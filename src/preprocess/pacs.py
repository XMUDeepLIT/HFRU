import io
import json
import os
import random
import re
import argparse
import multiprocessing as mp

import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

# Keep CUDA safe with vLLM worker processes; allow user-provided env to override.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm import LLM, SamplingParams


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ====== 固定配置 ======

REPLACE_PROMPT_TEMPLATE = (
    "Below is an image description. Correct ONLY grammar errors.\n"
    "Allowed edits are limited to basic grammar fixes such as:\n"
    "- article usage (a/an/the)\n"
    "- singular/plural agreement\n"
    "- subject-verb agreement\n"
    "Do NOT change meaning, wording choices, sentence order, style, or sentence structure.\n"
    "Do NOT add or remove content except what is strictly needed for grammar correctness.\n"
    "Output only the corrected description, nothing else.\n\nDescription: {description}"
)

LABELS = [
    "dog", "canine", "puppy", "hound", "shepherd", "terrier", "beagle", "mastiff", "mutt", "pooch",
    "pupper", "puppo", "pup", "mongrel", "tyke", "corgi", "poodle", "husky", "labrador", "beagle",
    "chihuahua", "pomeranian", "shiba", "samoyed", "dachshund", "collie", "rottweiler", "puppies",
    "huskies", "elephant", "mammoth"
]
LABEL_PATTERN = re.compile("|".join(re.escape(label) for label in LABELS), flags=re.IGNORECASE)
ANIMALS = ["cat", "rabbit", "hamster", "parrot", "fish", "mouse", "pig", "cow", "sheep", "chicken"]

TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"


def parse_args():
    parser = argparse.ArgumentParser(description="PACS preprocess pipeline")
    parser.add_argument("--model-path", default="./model/Qwen2.5-VL-3B-Instruct", help="Vision-language model path")
    parser.add_argument("--model-path2", default="./model/Qwen3-8B", help="Text model path for grammar fixing")
    parser.add_argument("--input-parquet", default="./data/dataset/pacs/data/train-00000-of-00001.parquet", help="Input parquet file path")
    parser.add_argument("--question", default="Please describe this image.", help="Question used for image description")
    parser.add_argument("--instruction", default="<image>Please describe this image.", help="Instruction stored in output records")
    parser.add_argument("--output-dir", default="./src/LlamaFactory/data", help="Output directory for LlamaFactory json/images")
    parser.add_argument("--model-name", default="qwen2.5_vl_3b", help="Model tag used in output file names")
    parser.add_argument(
        "--pacs-verl-dir",
        "--pacs-dir",
        dest="pacs_verl_dir",
        default="./data/pacs",
        help="Output directory for generated parquet files",
    )
    parser.add_argument("--data-source", default="flwrlabs/pacs", help="data_source field in parquet records")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=512, help="Generation max tokens")
    parser.add_argument("--max-model-len", type=int, default=4096, help="vLLM max model length")
    return parser.parse_args()


def build_runtime_config(args):
    model_name = args.model_name
    output_dir = args.output_dir
    pacs_verl_dir = args.pacs_verl_dir

    return {
        "model_path": args.model_path,
        "model_path2": args.model_path2,
        "input_parquet": args.input_parquet,
        "question": args.question,
        "instruction": args.instruction,
        "output_dir": output_dir,
        "model_name": model_name,
        "pacs_verl_dir": pacs_verl_dir,
        "data_source": args.data_source,
        "val_ratio": args.val_ratio,
        "random_seed": args.random_seed,
        "sampling_params": SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens),
        "max_model_len": args.max_model_len,
        "images_dir": os.path.join(output_dir, "pacs_images"),
        "json_path1_train": os.path.join(output_dir, "pacs_train.json"),
        "json_path1_val": os.path.join(output_dir, "pacs_val.json"),
        "json_path1_test": os.path.join(output_dir, "pacs_test.json"),
        "parquet_path_train": os.path.join(pacs_verl_dir, "train.parquet"),
        "parquet_path_val": os.path.join(pacs_verl_dir, "val.parquet"),
        "parquet_path_test": os.path.join(pacs_verl_dir, "test.parquet"),
    }


def ensure_output_dirs(cfg):
    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs(cfg["pacs_verl_dir"], exist_ok=True)


def load_dataframe(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    # 统一成连续唯一行号，避免后续按索引命名图片时发生覆盖。
    df = df.reset_index(drop=True)
    print(f"Loaded {len(df)} rows from parquet")
    return df


def parse_images(df: pd.DataFrame):
    images = []
    for img_dict in df["image"]:
        img = Image.open(io.BytesIO(img_dict["bytes"])).convert("RGB")
        images.append(img)
    return images


def build_vision_prompt(model_path: str, question: str) -> str:
    processor = AutoProcessor.from_pretrained(model_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_image_descriptions(images, prompt: str, model_path: str, max_model_len: int, sampling_params):
    llm = LLM(model=model_path, max_model_len=max_model_len)
    inputs = [
        {
            "prompt": prompt,
            "multi_modal_data": {"image": img},
        }
        for img in images
    ]
    print(f"Running inference on {len(inputs)} images...")
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    descriptions = [o.outputs[0].text for o in outputs]
    del llm
    return descriptions


def replace_labels_with_animals(descriptions):
    replaced_descriptions = []
    for desc in descriptions:
        replacement_animal = random.choice(ANIMALS)

        def replace_if_match(word_match: re.Match) -> str:
            word = word_match.group(0)
            return replacement_animal if LABEL_PATTERN.search(word) else word

        replaced_desc = re.sub(r"\b\w+\b", replace_if_match, desc)
        replaced_descriptions.append(replaced_desc)
    return replaced_descriptions


def grammar_fix_descriptions(descriptions, model_path2: str, max_model_len: int, sampling_params):
    tokenizer = AutoTokenizer.from_pretrained(model_path2)
    llm = LLM(model=model_path2, max_model_len=max_model_len)

    replace_messages_list = [
        [
            {
                "role": "user",
                "content": REPLACE_PROMPT_TEMPLATE.format(description=desc),
            }
        ]
        for desc in descriptions
    ]

    replace_prompts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for msgs in replace_messages_list
    ]
    replace_inputs = [{"prompt": p} for p in replace_prompts]

    print(f"Running replacement inference on {len(replace_inputs)} descriptions...")
    replace_outputs = llm.generate(replace_inputs, sampling_params=sampling_params)
    final_descriptions = [o.outputs[0].text for o in replace_outputs]
    del llm
    return final_descriptions


def split_by_domain(df: pd.DataFrame, val_ratio: float, random_seed: int):
    # 使用 domain 划分：sketch 作为测试集；cartoon/art_painting/photo 合并后再按 8:2 划分 train/val。
    if "domain" not in df.columns:
        raise KeyError("Expected 'domain' column in parquet data")

    eligible_domains = {"cartoon", "art_painting", "photo"}
    domain_series = df["domain"].astype(str).str.lower()

    test_row_ids = set(df[domain_series == "sketch"].index.tolist())
    train_val_candidate_df = df[domain_series.isin(eligible_domains)]

    candidate_indices = train_val_candidate_df.index.tolist()
    val_size = int(len(candidate_indices) * val_ratio)
    if len(candidate_indices) > 0 and val_size == 0:
        val_size = 1

    if val_size > 0:
        val_row_ids = set(train_val_candidate_df.sample(n=val_size, random_state=random_seed).index.tolist())
    else:
        val_row_ids = set()
    train_row_ids = set(candidate_indices) - val_row_ids

    unknown_row_ids = set(df.index.tolist()) - test_row_ids - set(candidate_indices)
    if unknown_row_ids:
        unknown_domains = sorted(df.loc[list(unknown_row_ids), "domain"].astype(str).unique().tolist())
        raise ValueError(f"Found unsupported domains: {unknown_domains}")

    print(
        f"Split with seed={random_seed}: "
        f"train={len(train_row_ids)}, val={len(val_row_ids)}, test(sketch)={len(test_row_ids)}"
    )
    return train_row_ids, val_row_ids, test_row_ids


def detect_image_ext(img_bytes: bytes) -> str:
    if img_bytes[:3] == b"\xff\xd8\xff":
        return "jpg"
    if img_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    return "jpg"


def save_json(path: str, records):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)


def save_outputs(records_by_split, parquet_records_by_split, cfg):
    save_json(cfg["json_path1_train"], records_by_split[TRAIN_SPLIT])
    save_json(cfg["json_path1_val"], records_by_split[VAL_SPLIT])
    save_json(cfg["json_path1_test"], records_by_split[TEST_SPLIT])

    pd.DataFrame(parquet_records_by_split[TRAIN_SPLIT]).to_parquet(cfg["parquet_path_train"], index=False)
    pd.DataFrame(parquet_records_by_split[VAL_SPLIT]).to_parquet(cfg["parquet_path_val"], index=False)
    pd.DataFrame(parquet_records_by_split[TEST_SPLIT]).to_parquet(cfg["parquet_path_test"], index=False)


def split_for_row(row_id: int, val_row_ids, test_row_ids) -> str:
    if row_id in test_row_ids:
        return TEST_SPLIT
    if row_id in val_row_ids:
        return VAL_SPLIT
    return TRAIN_SPLIT


def build_records(df: pd.DataFrame, val_row_ids, test_row_ids, instruction: str, images_dir: str, data_source: str):
    os.makedirs(images_dir, exist_ok=True)

    records_by_split = {TRAIN_SPLIT: [], VAL_SPLIT: [], TEST_SPLIT: []}
    parquet_records_by_split = {TRAIN_SPLIT: [], VAL_SPLIT: [], TEST_SPLIT: []}

    for row_id, row in df.iterrows():
        split = split_for_row(row_id, val_row_ids, test_row_ids)

        record1 = {
            "instruction": instruction,
            "input": "",
            "output": row["replaced_description"],
            "images": [],
        }

        img_bytes = row["image"]["bytes"]
        ext = detect_image_ext(img_bytes)
        img_filename = f"img{row_id}.{ext}"
        img_path = os.path.join(images_dir, img_filename)
        with open(img_path, "wb") as f:
            f.write(img_bytes)
        record1["images"].append(img_path)

        records_by_split[split].append(record1)

        parquet_record = {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": instruction}],
            "images": [row["image"]],
            "reward_model": {"style": "rule", "ground_truth": row["description"]},
            "extra_info": {
                "split": split,
                "index": int(row_id),
                "answer": row["label"],
                "domain": row.get("domain", None),
                "description": row["description"],
                "replaced_description": row["replaced_description"],
            },
        }
        parquet_records_by_split[split].append(parquet_record)

        if (row_id + 1) % 1000 == 0:
            print(f"Processed {row_id + 1}/{len(df)} rows")

    return records_by_split, parquet_records_by_split


def print_summary(records_by_split, cfg):
    print("Done! JSON saved to:")
    print(f"- {cfg['json_path1_train']}")
    print(f"- {cfg['json_path1_val']}")
    print(f"- {cfg['json_path1_test']}")
    print(f"Images saved to {cfg['images_dir']}/")
    print("Parquet saved to:")
    print(f"- {cfg['parquet_path_train']}")
    print(f"- {cfg['parquet_path_val']}")
    print(f"- {cfg['parquet_path_test']}")
    print(
        f"Total records: train={len(records_by_split[TRAIN_SPLIT])}, "
        f"val={len(records_by_split[VAL_SPLIT])}, test={len(records_by_split[TEST_SPLIT])}"
    )


def configure_multiprocessing_for_cuda():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        # This can happen if another component set the context before main().
        print(f"Warning: failed to set multiprocessing start method to spawn: {e}")


def main():
    configure_multiprocessing_for_cuda()
    args = parse_args()
    cfg = build_runtime_config(args)
    ensure_output_dirs(cfg)

    df = load_dataframe(cfg["input_parquet"])

    images = parse_images(df)
    prompt = build_vision_prompt(cfg["model_path"], cfg["question"])
    descriptions = generate_image_descriptions(
        images,
        prompt,
        cfg["model_path"],
        cfg["max_model_len"],
        cfg["sampling_params"],
    )
    df["description"] = descriptions

    replaced_descriptions = replace_labels_with_animals(descriptions)
    replaced_descriptions = grammar_fix_descriptions(
        replaced_descriptions,
        cfg["model_path2"],
        cfg["max_model_len"],
        cfg["sampling_params"],
    )
    df["replaced_description"] = replaced_descriptions

    train_row_ids, val_row_ids, test_row_ids = split_by_domain(df, cfg["val_ratio"], cfg["random_seed"])
    records_by_split, parquet_records_by_split = build_records(
        df,
        val_row_ids,
        test_row_ids,
        cfg["instruction"],
        cfg["images_dir"],
        cfg["data_source"],
    )

    save_outputs(records_by_split, parquet_records_by_split, cfg)
    print_summary(records_by_split, cfg)


if __name__ == "__main__":
    main()
