import os
import json
from typing import Any, Dict

import numpy as np
import torch
from datasets import load_from_disk
from tqdm import tqdm

from AttributionPipeline.config import CONFIG, TASKS
from AttributionPipeline.src.attribution_methods import AttributionMethod


def _ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def _extract_labels(raw_labels) -> np.ndarray:
    # Normalize labels to float array, handling nested lists
    return np.array([x[0] if isinstance(x, list) else x for x in raw_labels]).astype(float)


def run() -> None:

    method_name: str = CONFIG.get("method_name")
    task_index: int = int(CONFIG.get("task"))
    dataset_path: str = CONFIG.get("dataset_path")
    output_dir: str = CONFIG.get("output_dir")
    device: str = CONFIG.get("device", "cpu")
    dataset_part: str = CONFIG.get("dataset_part")
    batch_size: int = int(CONFIG.get("batch_size"))

    if not dataset_path:
        raise Exception

    print(f"Loading dataset from: {dataset_path}")
    ds = load_from_disk(dataset_path)

    # Prepare labels
    labels = _extract_labels(ds["labels"]) if "labels" in ds.column_names else None

    # Initialize attribution method (loads model+tokenizer internally)
    attr = AttributionMethod(
        method_name=method_name,
        task_index=task_index,
        label_index=0,
        model_dir=CONFIG.get("model_dir"),
        device=device,
    )

    _ensure_dir(output_dir)
    print(f"Writing attributions to: {output_dir}")

    num_samples = len(ds["input_ids"])
    records = []

    input_ids_list = ds["input_ids"]
    attention_mask_list = ds["attention_mask"]
    smiles_list = ds["smiles"] if "smiles" in ds.column_names else None

    with tqdm(total=num_samples, desc="Attributing & writing", unit="samples") as pbar:
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)

            batch_ids_py = []
            batch_mask_py = []
            for i in range(start, end):
                ids = input_ids_list[i]
                mask = attention_mask_list[i]
                if hasattr(ids, "tolist"):
                    ids = ids.tolist()
                if hasattr(mask, "tolist"):
                    mask = mask.tolist()
                batch_ids_py.append(ids)
                batch_mask_py.append(mask)

            input_ids = torch.tensor(batch_ids_py, dtype=torch.long, device=device)
            attention_mask = torch.tensor(batch_mask_py, dtype=torch.long, device=device)

            attributions, _ = attr.compute(
                input_ids=input_ids,
                attention_mask=attention_mask,
                n_steps=50,
                normalize=True,
            )

            # Per-sample processing within the batch
            for bi in range(end - start):
                ids = batch_ids_py[bi]
                mask = batch_mask_py[bi]

                seq_len = int(np.sum(mask))
                tokens = attr.tokenizer.convert_ids_to_tokens(ids[:seq_len])
                scores = attributions[bi, :seq_len].cpu().numpy().astype(float).tolist()

                if smiles_list is not None:
                    smiles = smiles_list[start + bi]
                else:
                    smiles = attr.tokenizer.decode([t for t, m in zip(ids, mask) if m == 1], skip_special_tokens=True)

                label_value = None
                if labels is not None:
                    try:
                        label_value = int(labels[start + bi])
                    except Exception:
                        label_value = float(labels[start + bi])

                records.append({
                    "smiles": smiles,
                    "tokens": tokens,
                    "scores": scores,
                    "label": label_value,
                    "method": method_name,
                })

            pbar.update(end - start)

    # Write a single JSON file named as dataset_part
    single_out_path = os.path.join(output_dir, f"{dataset_part}.json")
    with open(single_out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    print(f"Wrote single JSON file: {single_out_path}")


if __name__ == "__main__":
    run()


