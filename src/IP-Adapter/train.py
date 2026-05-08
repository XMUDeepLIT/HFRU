import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoProcessor,
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    Qwen2_5_VLForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)
from qwen_vl_utils import process_vision_info


logger = get_logger(__name__)


def preprocess_qwen_image(image, target_image_size):
    """Center-crop to square then resize for Qwen input."""
    width, height = image.size
    crop_size = min(width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size

    resampling = getattr(Image, "Resampling", Image)
    return image.crop((left, top, right, bottom)).resize(
        (target_image_size, target_image_size),
        resampling.LANCZOS,
    )


def load_clip_image_processor(clip_model_path):
    try:
        return CLIPImageProcessor.from_pretrained(clip_model_path)
    except OSError as error:
        config_path = Path(clip_model_path) / "config.json"
        image_size = 224
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                model_config = json.load(handle)
            image_size = int(model_config.get("image_size", image_size))

        logger.warning(
            "Failed to load CLIP image processor from %s (%s). "
            "Falling back to default CLIP preprocessing with image_size=%s.",
            clip_model_path,
            error,
            image_size,
        )

        resampling = getattr(Image, "Resampling", Image)
        return CLIPImageProcessor(
            do_resize=True,
            size={"shortest_edge": image_size},
            resample=resampling.BICUBIC,
            do_center_crop=True,
            crop_size={"height": image_size, "width": image_size},
            do_rescale=True,
            rescale_factor=1 / 255.0,
            do_normalize=True,
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
        )


def load_image_records(data_path):
    path = Path(data_path)
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        for key in ("data", "items", "records"):
            if key in payload and isinstance(payload[key], list):
                return payload[key]
        raise ValueError("Unsupported JSON structure. Expected a list or a dict with data/items/records.")

    if isinstance(payload, list):
        return payload

    raise ValueError("Unsupported dataset format.")


class ImageRecordDataset(Dataset):
    def __init__(self, data_file, image_root=""):
        self.records = load_image_records(data_file)
        self.image_root = image_root

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        if isinstance(record, str):
            image_file = record
        elif isinstance(record, dict):
            image_file = (
                record.get("image_file")
                or record.get("image")
                or record.get("path")
                or record.get("file")
            )
            if image_file is None:
                raise KeyError(
                    "Each dict record must contain one of: image_file, image, path, file. "
                    f"Got keys={list(record.keys())} at idx={idx}."
                )
        else:
            raise TypeError(
                "Each record must be either a string path or a dict. "
                f"Got type={type(record)} at idx={idx}."
            )

        image_path = image_file if os.path.isabs(image_file) else os.path.join(self.image_root, image_file)
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image).convert("RGB")
        return {"image": image, "path": image_path}


def collate_fn(batch):
    return {
        "images": [item["image"] for item in batch],
        "paths": [item["path"] for item in batch],
    }


class VisionToClipCrossAttentionProjector(torch.nn.Module):
    def __init__(self, in_dim, out_dim, num_queries, num_heads=8, ff_hidden_dim=None, dropout=0.0):
        super().__init__()
        if out_dim % num_heads != 0:
            raise ValueError(f"out_dim={out_dim} must be divisible by num_heads={num_heads}")

        ff_hidden_dim = ff_hidden_dim or max(in_dim, out_dim)
        self.num_queries = int(num_queries)
        self.in_proj = torch.nn.Linear(in_dim, out_dim)
        self.query_tokens = torch.nn.Parameter(torch.randn(self.num_queries, out_dim) * 0.02)
        self.query_ln = torch.nn.LayerNorm(out_dim)
        self.kv_ln = torch.nn.LayerNorm(out_dim)
        self.cross_attn = torch.nn.MultiheadAttention(
            embed_dim=out_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff_ln = torch.nn.LayerNorm(out_dim)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(out_dim, ff_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ff_hidden_dim, out_dim),
            torch.nn.Dropout(dropout),
        )
        self.out_ln = torch.nn.LayerNorm(out_dim)

    def forward(self, x, key_padding_mask=None):
        """
        Args:
            x: [batch, seq_len, in_dim] — variable-length Qwen vision features (padded).
            key_padding_mask: [batch, seq_len] — True for padded positions.
        Returns:
            [batch, num_queries, out_dim] — fixed-length output tokens.
        """
        kv = self.in_proj(x)
        kv = self.kv_ln(kv)
        queries = self.query_tokens.unsqueeze(0).expand(x.size(0), -1, -1)
        queries = self.query_ln(queries)
        attn_out, _ = self.cross_attn(
            query=queries, key=kv, value=kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        queries = queries + attn_out
        queries = queries + self.ff(self.ff_ln(queries))
        return self.out_ln(queries)


def build_parser():
    parser = argparse.ArgumentParser(description="Distill full Qwen2.5-VL output to CLIP hidden-token space for IP-Adapter Plus/PlusXL.")
    parser.add_argument("--qwen_model_path", type=str, default="./model/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--clip_model_path", type=str, default="./src/IP-Adapter/models/image_encoder")
    parser.add_argument("--data_file", type=str, default="./data/dataset/VGGFace2/train_image_paths.json")
    parser.add_argument("--image_root", type=str, default="./data/dataset/VGGFace2/train")
    parser.add_argument("--output_dir", type=str, default="mapper")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=0, help="If >0, use a fixed warmup step count; if <=0, derive warmup from warmup_ratio.")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio used when warmup_steps<=0.")
    parser.add_argument("--min_warmup_steps", type=int, default=10)
    parser.add_argument("--max_warmup_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--dataloader_num_workers", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mlp_hidden_dim", type=int, default=4096)
    parser.add_argument("--cross_attn_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--loss_cos_weight", type=float, default=1.0)
    parser.add_argument("--loss_global_weight", type=float, default=0.5)
    parser.add_argument("--loss_mse_weight", type=float, default=0.0, help="MSE loss weight. Default 0 (disabled) — cosine losses are more stable for feature alignment.")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--qwen_prompt", type=str, default="", help="Text prompt used to query the full Qwen2.5-VL model for each image.")
    parser.add_argument("--qwen_image_size", type=int, default=227, help="Center-crop and resize images to this square size before feeding them to Qwen.")
    parser.add_argument("--report_to", type=str, default="wandb", choices=["wandb", "tensorboard", "all", "none"], help="Tracker backend used by Accelerate.")
    parser.add_argument("--wandb_project", type=str, default="qwen25vl_clip_distill", help="WandB project name.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Optional WandB run name.")
    return parser


def infer_clip_seq_len(clip_model):
    vision_cfg = clip_model.config
    image_size = getattr(vision_cfg, "image_size", None)
    patch_size = getattr(vision_cfg, "patch_size", None)
    if image_size is None or patch_size is None:
        return 257
    return (int(image_size) // int(patch_size)) ** 2 + 1


def save_projector(output_dir, projector, qwen_model_path, clip_model_path, in_dim, out_dim, clip_seq_len, projector_type, cross_attn_heads):
    payload = {
        "projector": projector.state_dict(),
        "qwen_model_path": qwen_model_path,
        "clip_model_path": clip_model_path,
        "in_dim": in_dim,
        "out_dim": out_dim,
        "clip_seq_len": clip_seq_len,
        "projector_type": projector_type,
        "cross_attn_heads": cross_attn_heads,
        "source_type": "full_model",
    }
    torch.save(payload, os.path.join(output_dir, "qwen_to_clip_projector.pt"))


def load_qwen_full_model(qwen_model_path, weight_dtype, device):
    """Load the full Qwen2.5-VL model (vision encoder + language model).

    The entire model is frozen and used as a teacher for distillation.
    This uses more VRAM than vision-only loading but captures richer
    representations from the full vision-language pipeline.
    """
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        qwen_model_path,
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True,
    )
    qwen_model.to(device=device)
    qwen_model.requires_grad_(False)
    qwen_model.eval()
    return qwen_model


def prepare_qwen_inputs(qwen_processor, images, prompt_text="", target_image_size=227):
    """Prepare full model inputs for Qwen2.5-VL using chat template.

    Constructs a chat-format message with the image and a text prompt
    for each image, then processes them into model-ready tensors.

    Args:
        qwen_processor: AutoProcessor for Qwen2.5-VL.
        images: list of PIL.Image instances.
        prompt_text: text prompt to accompany each image.

    Returns:
        dict of batched tensors: input_ids, attention_mask, pixel_values, image_grid_thw, etc.
    """
    # Preprocess each image for Qwen: center-crop square + fixed resize.
    qwen_images = [preprocess_qwen_image(img, target_image_size) for img in images]

    # Build per-image chat messages
    messages_batch = []
    for img in qwen_images:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        messages_batch.append(messages)

    # Process each message through the chat template
    texts = []
    all_image_inputs = []
    for messages in messages_batch:
        text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts.append(text)
        image_inputs, _ = process_vision_info(messages)
        all_image_inputs.append(image_inputs)

    # Batch process
    inputs = qwen_processor(
        text=texts,
        images=all_image_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs


def extract_qwen_full_features(qwen_model, inputs, device, weight_dtype):
    """Extract last hidden state from the full Qwen2.5-VL model.

    Runs the full model forward pass (vision encoder + language model)
    and returns the last hidden state for each image in the batch.

    Args:
        qwen_model: Qwen2_5_VLForConditionalGeneration (frozen).
        inputs: dict of batched tensors from prepare_qwen_inputs().
        device: target device.
        weight_dtype: computation dtype.

    Returns:
        list of [seq_len_i, hidden_size] tensors — per-image LLM hidden states
        (padding removed).
    """
    # Move inputs to device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    pixel_values = inputs["pixel_values"].to(device, dtype=weight_dtype)
    image_grid_thw = inputs["image_grid_thw"].to(device)

    # Forward pass through full model
    outputs = qwen_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        output_hidden_states=True,
        return_dict=True,
    )

    # Extract last hidden state: [batch, seq_len, hidden_size]
    last_hidden_state = outputs.hidden_states[-1]

    # Split per-image, removing padding based on attention_mask
    per_image_features = []
    for i in range(last_hidden_state.size(0)):
        valid_len = attention_mask[i].sum().item()
        # print(f"Image {i}: valid_len={valid_len} seq_len={last_hidden_state.size(1)}")
        feat = last_hidden_state[i, :valid_len, :]  # [valid_len, hidden_size]
        per_image_features.append(feat)

    return per_image_features


def pad_and_batch_features(feature_list, dtype):
    """Pad variable-length per-image features to the same length and stack into a batch.

    Args:
        feature_list: list of [num_tokens_i, dim] tensors
        dtype: target dtype

    Returns:
        padded: [batch, max_len, dim]
        key_padding_mask: [batch, max_len] — True for padded positions
    """
    max_len = max(f.size(0) for f in feature_list)
    dim = feature_list[0].size(-1)
    batch_size = len(feature_list)
    device = feature_list[0].device

    padded = torch.zeros(batch_size, max_len, dim, dtype=dtype, device=device)
    key_padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool, device=device)  # True = masked

    for i, feat in enumerate(feature_list):
        seq_len = feat.size(0)
        padded[i, :seq_len, :] = feat.to(dtype=dtype)
        key_padding_mask[i, :seq_len] = False  # False = valid token

    return padded, key_padding_mask


def main():
    args = build_parser().parse_args()

    logging_dir = Path(args.output_dir, args.logging_dir)
    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    log_with = None if args.report_to == "none" else args.report_to
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=log_with,
        project_config=project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    qwen_processor = AutoProcessor.from_pretrained(args.qwen_model_path, use_fast=False)
    clip_processor = load_clip_image_processor(args.clip_model_path)
    accelerator.print("[init] processors loaded")

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load the full Qwen2.5-VL model (vision + language)
    qwen_model = load_qwen_full_model(
        args.qwen_model_path, weight_dtype, accelerator.device
    )
    accelerator.print("[init] full qwen2.5-vl model loaded")

    clip_model = CLIPVisionModelWithProjection.from_pretrained(
        args.clip_model_path,
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True,
    )
    clip_model.requires_grad_(False)
    clip_model.eval()
    clip_model.to(accelerator.device)
    accelerator.print("[init] clip model loaded and moved to device")

    # Use the LLM hidden size as projector input dimension
    qwen_hidden_dim = qwen_model.config.hidden_size
    clip_out_dim = clip_model.config.hidden_size
    clip_seq_len = infer_clip_seq_len(clip_model)

    accelerator.print(
        f"[init] qwen_hidden_dim(LLM)={qwen_hidden_dim} clip_out_dim={clip_out_dim} "
        f"clip_seq_len={clip_seq_len}"
    )

    projector = VisionToClipCrossAttentionProjector(
        in_dim=qwen_hidden_dim,
        out_dim=clip_out_dim,
        num_queries=clip_seq_len,
        num_heads=args.cross_attn_heads,
        ff_hidden_dim=args.mlp_hidden_dim,
        dropout=args.dropout,
    )

    if args.resume_from is not None:
        state = torch.load(args.resume_from, map_location="cpu")
        projector_type = state.get("projector_type", "mlp")
        if projector_type != "cross_attention":
            raise ValueError(
                f"resume_from expects a cross_attention checkpoint, got projector_type={projector_type}. "
                "Please start fresh or use a cross_attention projector checkpoint."
            )
        projector.load_state_dict(state["projector"], strict=True)

    optimizer = torch.optim.AdamW(projector.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    dataset = ImageRecordDataset(args.data_file, image_root=args.image_root)
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        timeout=600 if args.dataloader_num_workers > 0 else 0,
        collate_fn=collate_fn,
    )
    accelerator.print(f"[init] dataset size={len(dataset)} dataloader workers={args.dataloader_num_workers}")

    projector, optimizer, dataloader = accelerator.prepare(projector, optimizer, dataloader)
    accelerator.print("[init] accelerator.prepare finished")

    if accelerator.is_main_process and log_with is not None:
        init_kwargs = {}
        if args.report_to == "wandb" and args.wandb_run_name:
            init_kwargs["wandb"] = {"name": args.wandb_run_name}
        accelerator.init_trackers(
            args.wandb_project if args.report_to == "wandb" else Path(args.output_dir).name,
            config=vars(args),
            init_kwargs=init_kwargs if init_kwargs else None,
        )
        accelerator.print(f"[init] trackers initialized with {args.report_to}")

    num_update_steps_per_epoch = max(1, math.ceil(len(dataloader) / args.gradient_accumulation_steps))
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    auto_warmup_steps = int(max_train_steps * max(0.0, args.warmup_ratio))
    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else auto_warmup_steps
    warmup_steps = max(args.min_warmup_steps, warmup_steps)
    warmup_steps = min(args.max_warmup_steps, warmup_steps)
    warmup_steps = min(warmup_steps, max(0, max_train_steps - 1))
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )
    accelerator.print(
        f"[init] scheduler warmup_steps={warmup_steps} total_steps={max_train_steps} "
        f"(fixed={args.warmup_steps}, ratio={args.warmup_ratio})"
    )

    progress_bar = None
    if accelerator.is_main_process:
        progress_bar = tqdm(total=max_train_steps, desc="Training", dynamic_ncols=True)

    global_step = 0
    for epoch in range(args.num_train_epochs):
        projector.train()
        if epoch == 0:
            accelerator.print("[train] entering training loop")
        for step, batch in enumerate(dataloader):
            if epoch == 0 and step == 0:
                accelerator.print(f"[train] first batch loaded, global_batch={len(batch['images'])}")
            with accelerator.accumulate(projector):
                images = batch["images"]

                # ====== Extract features from both models (no grad) ======
                with torch.no_grad():
                    # --- Full Qwen2.5-VL forward pass (vision + LLM) ---
                    qwen_inputs = prepare_qwen_inputs(
                        qwen_processor,
                        images,
                        prompt_text=args.qwen_prompt,
                        target_image_size=args.qwen_image_size,
                    )
                    qwen_feature_list = extract_qwen_full_features(
                        qwen_model, qwen_inputs, accelerator.device, weight_dtype
                    )

                    # --- CLIP target features ---
                    clip_inputs = clip_processor(images=images, return_tensors="pt")
                    clip_pixel_values = clip_inputs["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                    target_tokens = clip_model(
                        pixel_values=clip_pixel_values,
                        output_hidden_states=True,
                    ).hidden_states[-2]  # [batch, 257, 1280]

                # ====== Project Qwen features ======
                # Cross-attention handles variable-length via key_padding_mask.
                qwen_padded, qwen_mask = pad_and_batch_features(qwen_feature_list, dtype=weight_dtype)
                pred_tokens = projector(qwen_padded, key_padding_mask=qwen_mask)
                # pred_tokens: [batch, clip_seq_len, clip_out_dim]

                # ====== Compute loss ======
                pred_tokens_f = pred_tokens.float()
                target_tokens_f = target_tokens.float()

                # Token-level cosine similarity loss (primary)
                loss_cos = 1.0 - F.cosine_similarity(pred_tokens_f, target_tokens_f, dim=-1).mean()

                # Global cosine similarity loss (sequence-level alignment)
                pred_global = pred_tokens_f.mean(dim=1)
                target_global = target_tokens_f.mean(dim=1)
                loss_global = 1.0 - F.cosine_similarity(pred_global, target_global, dim=-1).mean()

                loss = args.loss_cos_weight * loss_cos + args.loss_global_weight * loss_global

                # Optional MSE (disabled by default, can be enabled via --loss_mse_weight)
                if args.loss_mse_weight > 0:
                    # Normalize features before MSE to make it scale-invariant
                    pred_normed = F.normalize(pred_tokens_f, dim=-1)
                    target_normed = F.normalize(target_tokens_f, dim=-1)
                    loss_mse = F.mse_loss(pred_normed, target_normed)
                    loss = loss + args.loss_mse_weight * loss_mse
                else:
                    loss_mse = torch.tensor(0.0)

                # ====== Backward + Optimizer ======
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(projector.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if progress_bar is not None:
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        cos=f"{loss_cos.item():.4f}",
                        gcos=f"{loss_global.item():.4f}",
                        lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    )
                if accelerator.is_main_process and (
                    global_step == 1
                    or global_step % args.logging_steps == 0
                    or global_step == max_train_steps
                ):
                    accelerator.print(
                        f"[loss] epoch={epoch} step={global_step}/{max_train_steps} "
                        f"loss={loss.item():.6f} cos={loss_cos.item():.6f} "
                        f"gcos={loss_global.item():.6f} "
                        f"lr={optimizer.param_groups[0]['lr']:.6e}"
                    )
                    if log_with is not None:
                        log_values = {
                            "train/loss": loss.item(),
                            "train/loss_cos": loss_cos.item(),
                            "train/loss_global": loss_global.item(),
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "train/epoch": epoch,
                        }
                        if args.loss_mse_weight > 0:
                            log_values["train/loss_mse"] = loss_mse.item()
                        accelerator.log(log_values, step=global_step)
                if accelerator.is_main_process and global_step % args.save_steps == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    save_projector(
                        ckpt_dir,
                        accelerator.unwrap_model(projector),
                        args.qwen_model_path,
                        args.clip_model_path,
                        qwen_hidden_dim,
                        clip_out_dim,
                        clip_seq_len,
                        "cross_attention",
                        args.cross_attn_heads,
                    )

    if progress_bar is not None:
        progress_bar.close()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_projector(
            args.output_dir,
            accelerator.unwrap_model(projector),
            args.qwen_model_path,
            args.clip_model_path,
            qwen_hidden_dim,
            clip_out_dim,
            clip_seq_len,
            "cross_attention",
            args.cross_attn_heads,
        )

    if log_with is not None:
        accelerator.end_training()


if __name__ == "__main__":
    main()