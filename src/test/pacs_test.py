import argparse
import io
import json
import os
import re
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer


os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm import LLM, SamplingParams
from pacs_hallu import collect_question1_files, evaluate_one_file


QUESTION1 = "Please describe this image."
QUESTION2 = "Write a caption for this image in detail."
QUESTION3 = "Is there any {label} in the image?"
QUESTIONS = [
    ("question1", QUESTION1),
    ("question2", QUESTION2),
    ("question3", QUESTION3),
]

LABELS = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]

words = [
    ["dog", "canine", "puppy", "hound", "shepherd", "terrier", "beagle", "mastiff", "mutt", "pooch", "pupper", "puppo", "pup", "mongrel", "tyke", "corgi", "poodle", "husky", "labrador", "beagle", "chihuahua", "pomeranian", "shiba", "samoyed", "dachshund", "collie", "rottweiler", "puppies", "huskies"],
    ["elephant", "mammoth"],
    ["giraffe"],
    ["guitar", "instrument"],
    ["horse", "pony", "stallion", "mare", "foal", "colt", "filly", "mustang", "appaloosa", "thoroughbred", "steed", "equine", "ponies", "fillies"],
    ["house", "home", "residence", "dwelling", "abode", "habitation", "domicile", "place", "villia", "mansion", "apartment", "flat", "cottage", "cabin", "hut", "manor", "estate", "building", "room"],
    ["person", "human", "individual", "man", "woman", "child", "adult", "teenager", "kid", "guy", "gal", "friend", "neighbor", "stranger", "character", "someone", "somebody", "people", "men", "women", "figure"],
]


def extract_labels(dataframe: pd.DataFrame) -> pd.Series:
    if "label" in dataframe.columns:
        return dataframe["label"].astype(int)
    if "extra_info" in dataframe.columns:
        return dataframe["extra_info"].apply(
            lambda x: int(x.get("answer")) if isinstance(x, dict) and x.get("answer") is not None else -1
        )
    raise ValueError("Cannot find labels: expected 'label' or 'extra_info.answer'.")


def extract_image(entry):
    if isinstance(entry, dict) and "bytes" in entry:
        return Image.open(io.BytesIO(entry["bytes"])).convert("RGB")
    if isinstance(entry, (list, tuple, np.ndarray)) and len(entry) > 0:
        first = entry[0]
        if isinstance(first, dict) and "bytes" in first:
            return Image.open(io.BytesIO(first["bytes"])).convert("RGB")
    raise ValueError("Unsupported image format. Expected {'bytes': ...} or a list/array containing it.")


def model_dir_name(model_path: str) -> str:
    p = Path(model_path)
    model_name = p.parent.parent.parent.name if len(p.parents) >= 3 else p.name
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)


def run_single_question(
    llm: LLM,
    processor: AutoProcessor,
    images,
    labels: pd.Series,
    question_name: str,
    question,
    output_json_path: str,
    temperature: float,
    max_tokens: int,
):
    if isinstance(question, str):
        questions = [question] * len(images)
        question_for_log = question
    else:
        questions = list(question)
        if len(questions) != len(images):
            raise ValueError("Dynamic questions length must match number of images.")
        question_for_log = "<dynamic question per sample>"

    inputs = [
        {
            "prompt": processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": q},
                        ],
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            ),
            "multi_modal_data": {"image": img},
        }
        for img, q in zip(images, questions)
    ]

    print(f"Running inference on {len(inputs)} images with question: {question_for_log}")
    outputs = llm.generate(inputs, sampling_params=SamplingParams(temperature=temperature, max_tokens=max_tokens))
    descriptions = [o.outputs[0].text for o in outputs]

    counts = []
    for i, desc in enumerate(descriptions):
        label = labels.iloc[i]
        if label < 0 or label >= len(words):
            counts.append(0)
            continue

        count = 0
        for word in words[label]:
            count += len(re.findall(re.escape(word), desc, flags=re.IGNORECASE))
        counts.append(count)

    result_df = pd.DataFrame()
    result_df["answer"] = labels.astype(int).apply(lambda idx: LABELS[idx] if 0 <= idx < len(LABELS) else "unknown")
    result_df["description"] = descriptions
    result_df["count"] = counts
    result_df["score"] = np.where(labels > 1, (result_df["count"] > 0).astype(int), (result_df["count"] == 0).astype(int))

    forget_mask = labels <= 1
    retain_mask = labels > 1

    if question_name == "question3":
        yes_flags = result_df["description"].fillna("").str.contains(r"\byes\b", case=False, regex=True)
        result_df["score"] = yes_flags.astype(int)
        average_score = result_df["score"].mean()
        forget_score = result_df.loc[forget_mask, "score"].mean() if forget_mask.any() else float("nan")
        retain_score = result_df.loc[retain_mask, "score"].mean() if retain_mask.any() else float("nan")
    else:
        average_score = result_df["score"].mean()
        forget_score = result_df.loc[forget_mask, "score"].mean() if forget_mask.any() else float("nan")
        retain_score = result_df.loc[retain_mask, "score"].mean() if retain_mask.any() else float("nan")
    metrics = {
        "question": question_for_log,
        "question_name": question_name,
        "num_samples": int(len(result_df)),
        "average_score": float(average_score),
        "forget_score": float(forget_score),
        "retain_score": float(retain_score),
    }

    if question_name == "question3":
        yes_flags = result_df["description"].fillna("").str.contains(r"\byes\b", case=False, regex=True)
        yes_ratio_by_label = {}
        label_values = labels.astype(int).tolist()
        for idx, label_name in enumerate(LABELS):
            label_mask = np.array([label == idx for label in label_values])
            if label_mask.any():
                yes_ratio_by_label[label_name] = float(yes_flags[label_mask].mean())
        metrics["group1_dog_elephant"] = {
            "yes_count": int(yes_flags[forget_mask].sum()),
            "total": int(forget_mask.sum()),
            "yes_ratio": float(yes_flags[forget_mask].mean()) if forget_mask.any() else 0.0,
            "non_yes_ratio": float((~yes_flags[forget_mask]).mean()) if forget_mask.any() else 0.0,
        }
        metrics["group2_other_answers"] = {
            "yes_count": int(yes_flags[retain_mask].sum()),
            "total": int(retain_mask.sum()),
            "ratio": float(yes_flags[retain_mask].mean()) if retain_mask.any() else 0.0,
        }
        metrics["yes_ratio_by_label"] = yes_ratio_by_label
        metrics["yes_ratio_overall"] = float(yes_flags.mean())

    print(f"Average score: {average_score}")
    print(f"Forget score (labels <= 1): {forget_score}")
    print(f"Retain score (labels > 1): {retain_score}")

    result_df.to_json(output_json_path, orient="records", lines=True, force_ascii=False)
    metrics_path = str(Path(output_json_path).with_name(f"{Path(output_json_path).stem}_metrics.json"))
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved: {output_json_path}")
    print(f"Saved metrics: {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one user-provided model path on PACS parquet datasets.")
    parser.add_argument(
        "--input_parquets",
        type=str,
        nargs="+",
        default=[
            "./data/pacs/val.parquet",
            "./data/pacs/test.parquet",
        ],
        help="Input parquet paths. Supports multiple paths.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./model/Qwen2.5-VL-3B-Instruct",
        help="Path to one model directory (for example: .../global_step_xxx/actor/huggingface).",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./results/pacs",
        help="Root directory for saving evaluation outputs.",
    )
    parser.add_argument(
        "--hallu_model_path",
        type=str,
        default="./model/Qwen3-8B",
        help="Model path used by hallucination evaluator (text judge model).",
    )
    parser.add_argument(
        "--disable_hallu",
        action="store_true",
        help="Disable post-test hallucination evaluation.",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=4096,
        help="max_model_len for vLLM.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum generated tokens for each prompt.",
    )
    return parser.parse_args()


def get_model_path_from_user(args: argparse.Namespace) -> str:
    if args.model_path:
        model_path = args.model_path.strip()
    else:
        model_path = input("Please input model_path: ").strip()

    if not model_path:
        raise ValueError("model_path is empty.")

    if not Path(model_path).exists():
        raise ValueError(f"model_path does not exist: {model_path}")

    return model_path


def run_hallucination_eval(model_folder: str, hallu_model_path: str) -> None:
    model_folder_path = Path(model_folder)
    question1_files = collect_question1_files(model_folder_path)
    if not question1_files:
        print(f"No question1.jsonl found under: {model_folder_path}")
        return

    print(f"\nLoading hallucination evaluator model: {hallu_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(hallu_model_path)
    hallu_llm = LLM(model=hallu_model_path, max_model_len=4096)

    for input_path in question1_files:
        output_path = input_path.with_name("question1_prompt_eval_dog_elephant.jsonl")
        if output_path.exists():
            print(f"Skip hallucination eval (output exists): {output_path}")
            continue
        evaluate_one_file(input_path=input_path, output_path=output_path, llm=hallu_llm, tokenizer=tokenizer)

    del hallu_llm


def main() -> None:
    args = parse_args()
    model_path = get_model_path_from_user(args)
    output_root_dir = args.output_root
    hallu_model_path = args.hallu_model_path
    input_parquets = args.input_parquets

    os.makedirs(output_root_dir, exist_ok=True)

    datasets = []
    for input_parquet in input_parquets:
        parquet_path = Path(input_parquet)
        if not parquet_path.exists():
            print(f"Skipping missing parquet: {input_parquet}")
            continue

        print(f"\nLoading parquet: {input_parquet}")
        df = pd.read_parquet(input_parquet)
        labels = extract_labels(df)
        print(f"Loaded {len(df)} rows from parquet")

        image_column = "image" if "image" in df.columns else "images"
        images = [extract_image(image_entry) for image_entry in df[image_column]]
        datasets.append(
            {
                "name": parquet_path.stem,
                "images": images,
                "labels": labels,
            }
        )

    if not datasets:
        raise ValueError("No valid parquet datasets to evaluate.")

    model_folder = os.path.join(output_root_dir, model_dir_name(model_path))
    os.makedirs(model_folder, exist_ok=True)

    pending_questions = []
    for dataset in datasets:
        dataset_folder = os.path.join(model_folder, dataset["name"])
        os.makedirs(dataset_folder, exist_ok=True)
        for question_name, question_text in QUESTIONS:
            output_path = os.path.join(dataset_folder, f"{question_name}.jsonl")
            if not os.path.exists(output_path):
                pending_questions.append((dataset, question_name, question_text, output_path))

    if not pending_questions:
        print(f"\nSkipping model (all target files exist): {model_path}")
        if not args.disable_hallu:
            run_hallucination_eval(model_folder=model_folder, hallu_model_path=hallu_model_path)
        return

    print(f"\nLoading model: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)
    llm = LLM(model=model_path, max_model_len=args.max_model_len)

    for dataset, question_name, question_text, output_path in pending_questions:
        if os.path.exists(output_path):
            print(f"Skipping existing target file: {output_path}")
            continue

        labels = dataset["labels"]
        images = dataset["images"]

        if question_name == "question3":
            answer_indices = labels.astype(int).tolist()
            answer_names = [
                LABELS[idx] if 0 <= idx < len(LABELS) else "object"
                for idx in answer_indices
            ]
            question_input = [QUESTION3.format(label=name) for name in answer_names]
        else:
            question_input = question_text

        run_single_question(
            llm=llm,
            processor=processor,
            images=images,
            labels=labels,
            question_name=question_name,
            question=question_input,
            output_json_path=output_path,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

    del llm

    if not args.disable_hallu:
        run_hallucination_eval(model_folder=model_folder, hallu_model_path=hallu_model_path)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Start method may already be set in interactive/parent contexts.
        pass
    main()