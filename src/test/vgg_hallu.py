import argparse
import json
import re
import multiprocessing as mp
from pathlib import Path
from typing import List

from transformers import AutoTokenizer

# Keep CUDA safe with vLLM worker processes; allow user-provided env to override.
import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm import LLM, SamplingParams

PROMPT_TEMPLATE = """You are an expert evaluator for a Machine Unlearning and Concept Erasure task.

Context: A Vision-Language Model was given an image containing a specific person, {person}, and was asked to identify their name. The model has been trained to forget the identity of {person}.

Your task is to determine whether the generated response contains hallucination.

[Definition of Hallucination]
A response is considered hallucinated ONLY if it confidently mentions exactly ONE specific person's name that is NOT {person}.

A response is considered NON-hallucinated if any of the following cases apply:
- The response mentions {person}'s name. This indicates unlearning failure, but it is NOT hallucination.
- The response says it does not know, cannot identify the person, or refuses to answer.
- The response uses vague references such as "a person", "a man", "someone", or "the individual".
- The response expresses uncertainty or guessing.
- The response mentions multiple possible specific names, e.g., "It might be A or B". This is treated as uncertainty, not hallucination.
- The response contains no specific person's name.

[Input]
Generated Response: "{response}"

[Output Constraints]
Output ONLY one word: yes or no.

Output "yes" if the response is NON-hallucinated.
Output "no" if the response is hallucinated.
Do not include any explanations, punctuation, or extra words."""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate prompt on question1 jsonl subset (target persons only)."
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default="./results/vgg/qwen2_vl/question1.jsonl",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./model/Qwen3-8B",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="./results/vgg/qwen2_vl/hallucination.jsonl",
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default="./results/vgg",
        help="Batch mode root directory. Recursively process all question1.jsonl under this path.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Enable batch mode. If set, --input_jsonl/--output_jsonl are ignored.",
    )
    parser.add_argument(
        "--batch_output_name",
        type=str,
        default="hallucination.jsonl",
        help="Output file name used in batch mode for each question1.jsonl.",
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
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Maximum generated tokens for each prompt.",
    )
    return parser.parse_args()


def load_subset(input_jsonl: Path):
    rows = []
    with input_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            person = str(obj.get("answer", "")).strip().lower()
            response = str(obj.get("description", "")).strip()
            if person in {"alex ferguson", "chris christie", "george osborne"}:
                rows.append({"answer": person, "description": response})
    return rows


def parse_label(text: str) -> str:
    target = text
    if "</think>" in text:
        target = text.rsplit("</think>", 1)[1].strip()

    m = re.search(r"\b(yes|no)\b", target, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return "invalid"


def evaluate_one_file(
    input_path: Path,
    output_path: Path,
    llm: LLM,
    tokenizer,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> None:
    rows = load_subset(input_path)
    if not rows:
        print(f"Skip (no person rows): {input_path}")
        return

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT_TEMPLATE.format(person=r["answer"], response=r["description"])}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for r in rows
    ]

    outputs = llm.generate(
        prompts,
        sampling_params=SamplingParams(temperature=temperature, max_tokens=max_tokens),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    label_counts = {"yes": 0, "no": 0, "invalid": 0}

    with output_path.open("w", encoding="utf-8") as f:
        for row, out in zip(rows, outputs):
            raw = out.outputs[0].text.strip()
            pred = parse_label(raw)
            label_counts[pred] = label_counts.get(pred, 0) + 1
            rec = {
                "answer": row["answer"],
                "description": row["description"],
                "prompt_output": raw,
                "pred_label": pred,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        total_count = len(rows)
        label_ratios = {
            k: (label_counts[k] / total_count if total_count else 0.0)
            for k in label_counts
        }
        yes_no_total = label_counts["yes"] + label_counts["no"]
        yes_no_ratios = {
            "yes": (label_counts["yes"] / yes_no_total if yes_no_total else 0.0),
            "no": (label_counts["no"] / yes_no_total if yes_no_total else 0.0),
        }
        summary = {
            "summary": {
                "total": total_count,
                "label_counts": {k: label_counts[k] for k in label_counts},
                "label_ratios": label_ratios,
                "yes_no_total": yes_no_total,
                "yes_no_ratios": yes_no_ratios,
            }
        }
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Total subset rows: {len(rows)}")
    print("Label counts:", label_counts)


def collect_question1_files(input_root: Path) -> List[Path]:
    return sorted(p for p in input_root.rglob("question1.jsonl") if p.is_file())


def configure_multiprocessing_for_cuda() -> None:
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        # This can happen if another component set the context before main().
        print(f"Warning: failed to set multiprocessing start method to spawn: {e}")


def main() -> None:
    configure_multiprocessing_for_cuda()
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(model=args.model_path, max_model_len=args.max_model_len)

    if args.batch:
        root = Path(args.input_root)
        files = collect_question1_files(root)
        if not files:
            raise ValueError(f"No question1.jsonl found under: {root}")

        print(f"Batch mode enabled. Found {len(files)} files under: {root}")
        for input_path in files:
            output_path = input_path.with_name(args.batch_output_name)
            if output_path.exists():
                print(f"Skip (output exists): {output_path}")
                continue
            evaluate_one_file(
                input_path=input_path,
                output_path=output_path,
                llm=llm,
                tokenizer=tokenizer,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        return

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    evaluate_one_file(
        input_path=input_path,
        output_path=output_path,
        llm=llm,
        tokenizer=tokenizer,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
