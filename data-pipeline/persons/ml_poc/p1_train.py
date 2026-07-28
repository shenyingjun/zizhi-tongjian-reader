from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from core import score_spans
from p1_dataset import LABELS
from p1_windows import build_windows, labels_to_spans, merge_predictions


MODEL_NAME = "KoichiYasuoka/roberta-classical-chinese-base-char"


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _metric_payload(reference, prediction) -> dict:
    exact = score_spans(reference, prediction)
    overlap = score_spans(reference, prediction, overlap=True)
    return {
        "reference_spans": exact.reference,
        "prediction_spans": exact.predicted,
        "exact": {
            "true_positive": exact.true_positive,
            "precision": exact.precision,
            "recall": exact.recall,
            "f1": exact.f1,
        },
        "overlap_diagnostic": {
            "true_positive": overlap.true_positive,
            "precision": overlap.precision,
            "recall": overlap.recall,
            "f1": overlap.f1,
        },
    }


def evaluate(
    model,
    tokenizer,
    examples: list[dict],
    device,
    *,
    max_length: int,
    stride: int,
    batch_size: int,
) -> tuple[dict, list[dict]]:
    import torch

    model.eval()
    references = []
    predictions = []
    output_rows = []
    with torch.inference_mode():
        for example in examples:
            windows = build_windows(
                tokenizer,
                example,
                max_length=max_length,
                stride=stride,
            )
            predicted_ids = []
            for batch_start in range(0, len(windows), batch_size):
                batch = windows[batch_start:batch_start + batch_size]
                inputs = {
                    "input_ids": torch.tensor(
                        [row.input_ids for row in batch],
                        dtype=torch.long,
                        device=device,
                    ),
                    "attention_mask": torch.tensor(
                        [row.attention_mask for row in batch],
                        dtype=torch.long,
                        device=device,
                    ),
                }
                if batch[0].token_type_ids is not None:
                    inputs["token_type_ids"] = torch.tensor(
                        [row.token_type_ids for row in batch],
                        dtype=torch.long,
                        device=device,
                    )
                logits = model(**inputs).logits
                predicted_ids.extend(logits.argmax(dim=-1).cpu().tolist())
            labels, owned = merge_predictions(
                example["text"], windows, predicted_ids
            )
            reference = labels_to_spans(
                example,
                example["labels"],
                [char != "\n" for char in example["text"]],
            )
            prediction = labels_to_spans(example, labels, owned)
            references.extend(reference)
            predictions.extend(prediction)
            output_rows.append({
                "id": example["id"],
                "reference_spans": [
                    row.__dict__ for row in reference
                ],
                "prediction_spans": [
                    row.__dict__ for row in prediction
                ],
            })
    return _metric_payload(references, predictions), output_rows


def train(args: argparse.Namespace) -> dict:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the P1 timing run")
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    train_examples = _read_jsonl(args.dataset / "train.jsonl")
    dev_examples = _read_jsonl(args.dataset / "dev.jsonl")
    holdout_examples = _read_jsonl(args.dataset / "pilot_holdout.jsonl")
    flattened = []
    for example in train_examples:
        flattened.extend(build_windows(
            tokenizer,
            example,
            max_length=args.max_length,
            stride=args.stride,
        ))

    class WindowDataset(Dataset):
        def __len__(self):
            return len(flattened)

        def __getitem__(self, index):
            row = flattened[index]
            payload = {
                "input_ids": torch.tensor(row.input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(
                    row.attention_mask, dtype=torch.long
                ),
                "labels": torch.tensor(row.labels, dtype=torch.long),
            }
            if row.token_type_ids is not None:
                payload["token_type_ids"] = torch.tensor(
                    row.token_type_ids, dtype=torch.long
                )
            return payload

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        WindowDataset(),
        batch_size=args.micro_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    id2label = {index: label for index, label in enumerate(LABELS)}
    label2id = {label: index for index, label in id2label.items()}
    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )
    model.gradient_checkpointing_enable()
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(loader) / args.gradient_accumulation)
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, round(total_updates * args.warmup_ratio)),
        num_training_steps=total_updates,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    history = []
    optimizer.zero_grad(set_to_none=True)
    update = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for step, batch in enumerate(loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            (loss / args.gradient_accumulation).backward()
            epoch_loss += float(loss.detach())
            if (
                step % args.gradient_accumulation == 0
                or step == len(loader)
            ):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
        dev_metrics, _ = evaluate(
            model,
            tokenizer,
            dev_examples,
            device,
            max_length=args.max_length,
            stride=args.stride,
            batch_size=args.eval_batch_size,
        )
        history.append({
            "epoch": epoch,
            "mean_window_loss": epoch_loss / len(loader),
            "updates": update,
            "dev": dev_metrics,
        })
        print(json.dumps(history[-1], ensure_ascii=False))

    dev_metrics, dev_predictions = evaluate(
        model,
        tokenizer,
        dev_examples,
        device,
        max_length=args.max_length,
        stride=args.stride,
        batch_size=args.eval_batch_size,
    )
    holdout_metrics, holdout_predictions = evaluate(
        model,
        tokenizer,
        holdout_examples,
        device,
        max_length=args.max_length,
        stride=args.stride,
        batch_size=args.eval_batch_size,
    )
    elapsed = time.perf_counter() - started
    report = {
        "schema_version": 1,
        "model": args.model,
        "plain_challenger": True,
        "self_agreement": "waived; references provisionally trusted",
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "config": {
            "epochs": args.epochs,
            "max_length": args.max_length,
            "stride": args.stride,
            "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "effective_batch_size": (
                args.micro_batch_size * args.gradient_accumulation
            ),
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "seed": args.seed,
            "precision": "fp32",
        },
        "timing": {
            "train_and_evaluate_seconds": elapsed,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
            "train_windows": len(flattened),
            "optimizer_updates": total_updates,
        },
        "history": history,
        "dev_challenge": dev_metrics,
        "random_pilot_holdout": holdout_metrics,
        "claim_limit": "pilot evidence only; holdout is not a sealed test",
    }
    model.save_pretrained(args.output / "model")
    tokenizer.save_pretrained(args.output / "model")
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for name, rows in (
        ("dev_predictions.json", dev_predictions),
        ("holdout_predictions.json", holdout_predictions),
    ):
        (args.output / name).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the plain P1 char-BIO challenger."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()
    report = train(args)
    print(json.dumps({
        "timing": report["timing"],
        "dev_challenge": report["dev_challenge"]["exact"],
        "random_pilot_holdout": report["random_pilot_holdout"]["exact"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
