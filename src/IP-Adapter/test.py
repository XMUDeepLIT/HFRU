import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from ip_adapter import IPAdapterPlusXL
from ip_adapter.custom_pipelines import StableDiffusionXLCustomPipeline


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_image_paths(image_path=None, image_dir=None):
    """Collect input image paths from a single file or a directory."""
    if image_dir:
        img_dir = Path(image_dir)
        if not img_dir.exists() or not img_dir.is_dir():
            raise ValueError(f"image_dir does not exist or is not a directory: {image_dir}")

        image_paths = sorted(
            str(p) for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise ValueError(f"No image files found in directory: {image_dir}")
        return image_paths

    if not image_path:
        raise ValueError("Either --image_path or --image_dir must be provided")

    if not os.path.isfile(image_path):
        raise ValueError(f"image_path does not exist: {image_path}")
    return [image_path]


def resolve_output_path(output_path, input_image_path, use_batch_mode):
    """Resolve output file path for single-image and batch-image inference."""
    input_stem = Path(input_image_path).stem
    output_obj = Path(output_path)

    if not use_batch_mode:
        return str(output_obj)

    # Batch mode: if output_path looks like a file, use its parent as output dir and
    # prepend the file stem to generated filenames.
    if output_obj.suffix:
        output_dir = output_obj.parent if str(output_obj.parent) else Path(".")
        prefix = output_obj.stem
    else:
        output_dir = output_obj
        prefix = ""

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{prefix}_{input_stem}.png" if prefix else f"{input_stem}.png"
    return str(output_dir / filename)


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


# ============================================================
# Projector (same architecture as in train.py)
# ============================================================

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


# ============================================================
# Full Qwen2.5-VL model loading (same as train.py)
# ============================================================

def load_qwen_full_model(qwen_model_path, weight_dtype, device):
    """Load the full Qwen2.5-VL model (vision encoder + language model)."""
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
    """Prepare full model inputs for Qwen2.5-VL using chat template."""
    qwen_images = [preprocess_qwen_image(img, target_image_size) for img in images]

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

    texts = []
    all_image_inputs = []
    for messages in messages_batch:
        text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts.append(text)
        image_inputs, _ = process_vision_info(messages)
        all_image_inputs.append(image_inputs)

    inputs = qwen_processor(
        text=texts,
        images=all_image_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs


def extract_qwen_full_features(qwen_model, inputs, device, weight_dtype):
    """Extract last hidden state from the full Qwen2.5-VL model."""
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    pixel_values = inputs["pixel_values"].to(device, dtype=weight_dtype)
    image_grid_thw = inputs["image_grid_thw"].to(device)

    outputs = qwen_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        output_hidden_states=True,
        return_dict=True,
    )

    last_hidden_state = outputs.hidden_states[-1]

    per_image_features = []
    for i in range(last_hidden_state.size(0)):
        valid_len = attention_mask[i].sum().item()
        feat = last_hidden_state[i, :valid_len, :]
        per_image_features.append(feat)

    return per_image_features


def pad_and_batch_features(feature_list, dtype):
    """Pad variable-length features and stack into a batch."""
    max_len = max(f.size(0) for f in feature_list)
    dim = feature_list[0].size(-1)
    batch_size = len(feature_list)
    device = feature_list[0].device

    padded = torch.zeros(batch_size, max_len, dim, dtype=dtype, device=device)
    key_padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool, device=device)

    for i, feat in enumerate(feature_list):
        seq_len = feat.size(0)
        padded[i, :seq_len, :] = feat.to(dtype=dtype)
        key_padding_mask[i, :seq_len] = False

    return padded, key_padding_mask


# ============================================================
# Load trained projector from checkpoint
# ============================================================

def load_projector(projector_path, device, dtype=torch.float16):
    """Load the trained projector from a checkpoint."""
    state = torch.load(projector_path, map_location="cpu")

    in_dim = state["in_dim"]
    out_dim = state["out_dim"]
    clip_seq_len = state["clip_seq_len"]
    cross_attn_heads = state.get("cross_attn_heads", 8)
    source_type = state.get("source_type", "vision_only")

    # Infer ff_hidden_dim from checkpoint weights if not saved as metadata.
    ff_hidden_dim = state.get("ff_hidden_dim", None)
    if ff_hidden_dim is None:
        proj_sd = state["projector"]
        if "ff.0.weight" in proj_sd:
            ff_hidden_dim = proj_sd["ff.0.weight"].shape[0]
        else:
            ff_hidden_dim = max(in_dim, out_dim)

    print(f"[projector] in_dim={in_dim} out_dim={out_dim} "
          f"clip_seq_len={clip_seq_len} heads={cross_attn_heads} "
          f"ff_hidden_dim={ff_hidden_dim} source_type={source_type}")

    projector = VisionToClipCrossAttentionProjector(
        in_dim=in_dim,
        out_dim=out_dim,
        num_queries=clip_seq_len,
        num_heads=cross_attn_heads,
        ff_hidden_dim=ff_hidden_dim,
    )
    projector.load_state_dict(state["projector"])
    projector.to(device=device, dtype=dtype)
    projector.eval()

    return projector, state


# ============================================================
# Modified IPAdapterPlusXL that uses full Qwen2.5-VL + Projector
# ============================================================

class IPAdapterPlusXLQwen(IPAdapterPlusXL):
    """IPAdapterPlusXL with full Qwen2.5-VL model replacing CLIP.

    The pipeline becomes:
        PIL Image → Qwen processor → Full Qwen2.5-VL (vision + LLM)
        → Trained projector (outputs CLIP-like hidden states)
        → IP-Adapter Resampler → UNet cross-attention
    """

    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, device,
                 num_tokens=16,
                 qwen_model=None, qwen_processor=None, projector=None,
                 clip_hidden_size=None, qwen_prompt="",
                 qwen_image_size=227):
        # Store Qwen components before parent __init__
        self._qwen_model = qwen_model
        self._qwen_processor = qwen_processor
        self._projector = projector
        self._clip_hidden_size = clip_hidden_size
        self._qwen_prompt = qwen_prompt
        self._qwen_image_size = qwen_image_size

        # Call parent __init__
        super().__init__(sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens)

        # Store as proper attributes
        self.qwen_model = qwen_model
        self.qwen_processor = qwen_processor
        self.projector = projector
        self.qwen_prompt = qwen_prompt
        self.qwen_image_size = qwen_image_size

    @torch.inference_mode()
    def get_image_embeds(self, pil_image):
        """Replace CLIP encoding with full Qwen2.5-VL + projector.

        Flow:
            1. Qwen processor → full model inputs (input_ids, pixel_values, etc.)
            2. Full Qwen2.5-VL → LLM last hidden states [seq_len, hidden_size]
            3. Pad + batch → [batch, max_len, hidden_size]
            4. Projector → [batch, 257, 1280]  (same as CLIP hidden_states[-2])
            5. IP-Adapter Resampler → [batch, num_tokens, cross_attn_dim]
        """
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]

        # === Full Qwen2.5-VL encoding (vision + LLM) ===
        qwen_inputs = prepare_qwen_inputs(
            self.qwen_processor,
            pil_image,
            prompt_text=self.qwen_prompt,
            target_image_size=self.qwen_image_size,
        )
        dtype = next(self.qwen_model.parameters()).dtype
        qwen_feature_list = extract_qwen_full_features(
            self.qwen_model, qwen_inputs, self.device, dtype
        )

        # === Projector: Qwen LLM features → CLIP-like hidden states ===
        qwen_padded, qwen_mask = pad_and_batch_features(qwen_feature_list, dtype=dtype)
        clip_like_embeds = self.projector(qwen_padded, key_padding_mask=qwen_mask)
        # clip_like_embeds: [batch, 257, 1280] — same format as CLIP hidden_states[-2]

        # === IP-Adapter Resampler ===
        image_prompt_embeds = self.image_proj_model(clip_like_embeds)

        # === Unconditional: zero input through projector + resampler ===
        zero_features = torch.zeros_like(qwen_padded)
        uncond_clip_like_embeds = self.projector(zero_features)
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_like_embeds)

        return image_prompt_embeds, uncond_image_prompt_embeds


# ============================================================
# Utility
# ============================================================

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Test IP-Adapter PlusXL with full Qwen2.5-VL model.")
    parser.add_argument("--projector_path", type=str, required=True,
                        help="Path to trained projector checkpoint (qwen_to_clip_projector.pt)")
    parser.add_argument("--qwen_model_path", type=str, default="./model/Qwen2.5-VL-3B-Instruct",
                        help="Path to Qwen2.5-VL model (full model will be loaded)")
    parser.add_argument("--base_model_path", type=str,
                        default="./model/stable-diffusion-xl-base-1.0",
                        help="Path to SDXL base model")
    parser.add_argument("--image_encoder_path", type=str, default="./models/image_encoder",
                        help="Path to CLIP image encoder (needed for Resampler config and IP-Adapter weights)")
    parser.add_argument("--ip_ckpt", type=str, default="sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin",
                        help="Path to IP-Adapter checkpoint")
    parser.add_argument("--image_path", type=str, default=None,
                        help="Single input image path (optional when --image_dir is provided)")
    parser.add_argument("--image_dir", type=str,
                        default=None,
                        help="Input image directory. All supported images will be processed.")
    parser.add_argument("--output_path", type=str, default="results/test.png",
                        help="Output image path")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Optional text prompt")
    parser.add_argument("--negative_prompt", type=str, default=None,
                        help="Optional negative prompt")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of images to generate")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_tokens", type=int, default=16,
                        help="Number of IP-Adapter tokens")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="IP-Adapter conditioning scale")
    parser.add_argument("--compare", action="store_true",
                        help="Also generate with original CLIP encoder for comparison")
    parser.add_argument("--qwen_prompt", type=str, default="",
                        help="Text prompt used to query the full Qwen2.5-VL model for each image.")
    parser.add_argument("--qwen_image_size", type=int, default=227,
                        help="Center-crop and resize images to this square size before feeding them to Qwen.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda"
    dtype = torch.float16

    input_image_paths = collect_image_paths(args.image_path, args.image_dir)
    batch_mode = len(input_image_paths) > 1

    print(f"[input] Found {len(input_image_paths)} image(s) to process")

    # === Load SDXL pipeline ===
    print("[1/5] Loading SDXL pipeline...")
    pipe = StableDiffusionXLCustomPipeline.from_pretrained(
        args.base_model_path,
        torch_dtype=dtype,
        add_watermarker=False,
    )

    # === Load full Qwen2.5-VL model ===
    print("[2/5] Loading full Qwen2.5-VL model...")
    qwen_model = load_qwen_full_model(args.qwen_model_path, dtype, device)
    qwen_processor = AutoProcessor.from_pretrained(args.qwen_model_path, use_fast=False)

    # === Load trained projector ===
    print("[3/5] Loading trained projector...")
    projector, proj_state = load_projector(args.projector_path, device, dtype)

    # === Create IP-Adapter with Qwen encoder ===
    print("[4/5] Initializing IP-Adapter PlusXL with full Qwen2.5-VL model...")
    ip_model = IPAdapterPlusXLQwen(
        sd_pipe=pipe,
        image_encoder_path=args.image_encoder_path,
        ip_ckpt=args.ip_ckpt,
        device=device,
        num_tokens=args.num_tokens,
        qwen_model=qwen_model,
        qwen_processor=qwen_processor,
        projector=projector,
        clip_hidden_size=proj_state["out_dim"],
        qwen_prompt=args.qwen_prompt,
        qwen_image_size=args.qwen_image_size,
    )

    # Free CLIP encoder from memory since we don't need it for inference
    del ip_model.image_encoder
    del ip_model.clip_image_processor
    torch.cuda.empty_cache()
    print("[4/5] CLIP encoder freed from memory")

    # === Generate ===
    print(f"[5/5] Generating {args.num_samples} images per input...")
    for idx, image_path in enumerate(input_image_paths, start=1):
        print(f"[gen {idx}/{len(input_image_paths)}] {image_path}")
        image = Image.open(image_path).convert("RGB")

        images = ip_model.generate(
            pil_image=image,
            num_samples=args.num_samples,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed
        )

        output_file = resolve_output_path(args.output_path, image_path, batch_mode)
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

        # Save results
        if len(images) > 1:
            grid = image_grid(images, 1, len(images))
            grid.save(output_file)
            print(f"Saved grid ({len(images)} images) to {output_file}")
        else:
            images[0].save(output_file)
            print(f"Saved image to {output_file}")

    # === Optional: Compare with original CLIP ===
    if args.compare:
        print("\n[compare] Generating with original CLIP encoder for comparison...")
        ip_model_clip = IPAdapterPlusXL(
            pipe, args.image_encoder_path, args.ip_ckpt, device, num_tokens=args.num_tokens
        )
        for idx, image_path in enumerate(input_image_paths, start=1):
            print(f"[compare {idx}/{len(input_image_paths)}] {image_path}")
            image = Image.open(image_path).convert("RGB")

            images_qwen = ip_model.generate(
                pil_image=image,
                num_samples=args.num_samples,
                num_inference_steps=args.num_inference_steps,
                seed=args.seed
            )
            images_clip = ip_model_clip.generate(
                pil_image=image,
                num_samples=args.num_samples,
                num_inference_steps=args.num_inference_steps,
                seed=args.seed
            )

            base_output = resolve_output_path(args.output_path, image_path, batch_mode)
            base_output_obj = Path(base_output)
            compare_path = str(base_output_obj.with_name(f"{base_output_obj.stem}_compare.png"))

            # Create side-by-side: top row Qwen, bottom row CLIP
            all_images = images_qwen + images_clip
            grid = image_grid(all_images, 2, len(images_qwen))
            grid.save(compare_path)
            print(f"Saved comparison grid to {compare_path}")
            print("  Top row: Qwen encoder | Bottom row: CLIP encoder")


if __name__ == "__main__":
    main()
