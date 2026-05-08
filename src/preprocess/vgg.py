import argparse
import io
import json
import os
import multiprocessing as mp
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoProcessor

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

TRAIN_SPLIT = "train"
TEST_SPLIT = "test"


def normalize_split(raw_split: str) -> str:
    split = str(raw_split).strip().lower()
    if split in {"train", "training"}:
        return TRAIN_SPLIT
    if split in {"test", "testing"}:
        return TEST_SPLIT
    raise ValueError(f"Unsupported split value: {raw_split}")


def extract_image_bytes(image_obj: object) -> bytes:
    if isinstance(image_obj, dict) and "bytes" in image_obj:
        return image_obj["bytes"]
    if isinstance(image_obj, (bytes, bytearray)):
        return bytes(image_obj)
    raise TypeError(f"Unsupported image object type: {type(image_obj)}")


def detect_ext(img_bytes: bytes) -> str:
    if img_bytes[:3] == b"\xff\xd8\xff":
        return "jpg"
    if img_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    return "jpg"


def preprocess_image_bytes(img_bytes: bytes, ext: str, target_image_size: int) -> bytes:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    width, height = img.size
    crop_size = min(width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size

    img = img.crop((left, top, right, bottom)).resize(
        (target_image_size, target_image_size), Image.Resampling.LANCZOS
    )

    output = io.BytesIO()
    save_format = "PNG" if ext == "png" else "JPEG"
    img.save(output, format=save_format)
    return output.getvalue()


def save_json(path: str, data: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess VGGFace2 parquet and build outputs.")
    parser.add_argument("--model-path", default="./model/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--input-parquet", default="./data/dataset/vgg/vgg_select.parquet")
    parser.add_argument("--question", default="What's the name of the person in this image?")
    parser.add_argument("--instruction", default="<image>What's the name of the person in this image?")
    parser.add_argument("--output-dir", default="./src/LlamaFactory/data")
    parser.add_argument("--parquet-output-dir", default="./data/vgg")
    parser.add_argument("--dataset-name", default="vgg")
    parser.add_argument("--data-source", default="VGGFace2")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=0)  # 0 表示使用全部样本
    parser.add_argument("--target-image-size", type=int, default=227)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--target-words",
        default="ferguson,christie,osborne",
        help="Comma-separated keywords for replacement filtering.",
    )
    return parser.parse_args()


def parse_target_words(raw: str) -> tuple[str, ...]:
    words = tuple(w.strip().lower() for w in raw.split(",") if w.strip())
    if not words:
        raise ValueError("--target-words must include at least one word")
    return words


def load_dataframe(input_parquet: str, max_samples: int) -> pd.DataFrame:
    df = pd.read_parquet(input_parquet).reset_index(drop=True)
    if max_samples > 0:
        df = df.head(max_samples).copy()
        print(f"Using first {len(df)} rows due to MAX_SAMPLES={max_samples}")

    required_cols = {"image", "split", "label"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise KeyError(f"Missing required columns: {sorted(missing_cols)}")

    print(f"Loaded {len(df)} rows from parquet")
    return df
def build_prompt(model_path: str, question: str) -> str:
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
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_descriptions(df: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    prompt = build_prompt(args.model_path, args.question)
    llm = LLM(model=args.model_path, max_model_len=args.max_model_len)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    images = [
        Image.open(io.BytesIO(extract_image_bytes(image_obj))).convert("RGB")
        for image_obj in df["image"]
    ]
    inputs = [
        {
            "prompt": prompt,
            "multi_modal_data": {"image": img},
        }
        for img in images
    ]

    print(f"Running description generation on {len(inputs)} images...")
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    descriptions = [o.outputs[0].text for o in outputs]

    del llm
    return descriptions

def contains_target_word(desc: str, target_words: tuple[str, ...]) -> bool:
    lower_desc = desc.lower()
    return any(word in lower_desc for word in target_words)


def build_replaced_descriptions(
    descriptions: list[str],
    target_words: tuple[str, ...],
    random_seed: int,
) -> list[str]:
    non_target_descriptions = [
        desc for desc in descriptions if not contains_target_word(desc, target_words)
    ]

    if not non_target_descriptions:
        print("Warning: no non-target descriptions found; keeping original descriptions.")
        return descriptions.copy()

    rng = np.random.default_rng(random_seed)
    replaced_descriptions = []
    for desc in descriptions:
        if contains_target_word(desc, target_words):
            replaced_descriptions.append(str(rng.choice(non_target_descriptions)))
        else:
            replaced_descriptions.append(desc)
    return replaced_descriptions


def prepare_output_paths(
    output_dir: str,
    parquet_output_dir: str,
    dataset_name: str,
) -> dict[str, str]:
    images_dir = os.path.join(output_dir, "vgg_images")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(parquet_output_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    return {
        "images_dir": images_dir,
        "json_path1_train": os.path.join(output_dir, f"{dataset_name}_train.json"),
        "json_path1_test": os.path.join(output_dir, f"{dataset_name}_test.json"),
        "parquet_path_train": os.path.join(
            parquet_output_dir, "train.parquet"
        ),
        "parquet_path_test": os.path.join(
            parquet_output_dir, "test.parquet"
        ),
    }


def build_output_records(
    df: pd.DataFrame,
    images_dir: str,
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, int],
]:
    records1_train: list[dict[str, Any]] = []
    records1_test: list[dict[str, Any]] = []
    parquet_records_train: list[dict[str, Any]] = []
    parquet_records_test: list[dict[str, Any]] = []
    split_counts = {TRAIN_SPLIT: 0, TEST_SPLIT: 0}

    for row_id, row in df.iterrows():
        split = normalize_split(row["split"])
        label = int(row["label"])
        description = str(row["description"])
        replaced_description = str(row["replaced_description"])

        original_img_bytes = extract_image_bytes(row["image"])
        ext = detect_ext(original_img_bytes)
        img_bytes = preprocess_image_bytes(original_img_bytes, ext, args.target_image_size)
        img_filename = f"{args.dataset_name}_{row_id}.{ext}"
        img_path = os.path.join(images_dir, img_filename)
        with open(img_path, "wb") as f:
            f.write(img_bytes)

        record1 = {
            "instruction": args.instruction,
            "input": "",
            "output": replaced_description,
            "images": [img_path],
        }

        parquet_record = {
            "data_source": args.data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": args.instruction,
                }
            ],
            "images": [{"bytes": img_bytes}],
            "reward_model": {"style": "rule", "ground_truth": description},
            "extra_info": {
                "split": split,
                "index": int(row_id),
                "answer": label,
                "description": description,
                "replaced_description": replaced_description,
            },
        }

        if split == TRAIN_SPLIT:
            records1_train.append(record1)
            parquet_records_train.append(parquet_record)
        else:
            records1_test.append(record1)
            parquet_records_test.append(parquet_record)

        split_counts[split] += 1

        if (row_id + 1) % 1000 == 0:
            print(f"Processed {row_id + 1}/{len(df)} rows")

    return (
        records1_train,
        records1_test,
        parquet_records_train,
        parquet_records_test,
        split_counts,
    )


def save_outputs(
    paths: dict[str, str],
    records1_train: list[dict[str, Any]],
    records1_test: list[dict[str, Any]],
    parquet_records_train: list[dict[str, Any]],
    parquet_records_test: list[dict[str, Any]],
    split_counts: dict[str, int],
) -> None:
    save_json(paths["json_path1_train"], records1_train)
    save_json(paths["json_path1_test"], records1_test)

    pd.DataFrame(parquet_records_train).to_parquet(paths["parquet_path_train"], index=False)
    pd.DataFrame(parquet_records_test).to_parquet(paths["parquet_path_test"], index=False)

    print("Done! JSON saved to:")
    print(f"- {paths['json_path1_train']}")
    print(f"- {paths['json_path1_test']}")
    print("Parquet saved to:")
    print(f"- {paths['parquet_path_train']}")
    print(f"- {paths['parquet_path_test']}")
    print(f"Images saved to {paths['images_dir']}/")
    print(
        "Total records: "
        f"train={split_counts[TRAIN_SPLIT]}, test={split_counts[TEST_SPLIT]}"
    )


def configure_multiprocessing_for_cuda() -> None:
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        # This can happen if another component set the context before main().
        print(f"Warning: failed to set multiprocessing start method to spawn: {e}")


def main() -> None:
    configure_multiprocessing_for_cuda()
    args = parse_args()
    target_words = parse_target_words(args.target_words)

    df = load_dataframe(args.input_parquet, args.max_samples)
    df["description"] = generate_descriptions(df, args)
    df["replaced_description"] = build_replaced_descriptions(
        df["description"].tolist(),
        target_words,
        args.random_seed,
    )

    paths = prepare_output_paths(
        args.output_dir,
        args.parquet_output_dir,
        args.dataset_name,
    )
    (
        records1_train,
        records1_test,
        parquet_records_train,
        parquet_records_test,
        split_counts,
    ) = build_output_records(df, paths["images_dir"], args)

    save_outputs(
        paths,
        records1_train,
        records1_test,
        parquet_records_train,
        parquet_records_test,
        split_counts,
    )


if __name__ == "__main__":
    main()
