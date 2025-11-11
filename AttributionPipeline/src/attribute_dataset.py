import os
import json
from typing import Any, Dict, List

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
    return np.array([x[0] if isinstance(x, list) else x for x in raw_labels]).astype(float)

#
# Split a SMILES token into units while preserving common two-character atoms.
# This keeps occurrences of 'Cl' and 'Br' together as one unit; everything else
# falls back to single-character splits.
#
_TWO_CHAR_ATOMS: set = {
    "He", "Li", "Be", "Ne", "Na", "Mg", "Al", "Si", "Cl", "Ar", "Ca",
    "Sc", "Ti", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
    "As", "Se", "Br", "Kr", "Rb", "Sr", "Zr", "Nb", "Mo", "Tc", "Ru",
    "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "Xe", "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
    "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "Re", "Os", "Ir", "Pt", "Au",
    "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No",
    "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Fl",
    "Lv", "Ts", "Og"
}

def _split_smiles_preserving_atoms(token: str) -> List[str]:
    units: List[str] = []
    i = 0
    while i < len(token):
        if i + 1 < len(token):
            pair = token[i : i + 2]
            if pair in _TWO_CHAR_ATOMS:
                units.append(pair)
                i += 2
                continue
        units.append(token[i])
        i += 1
    return units


def run() -> None:
    method_name: str = CONFIG.get("method_name")
    task_index: int = int(CONFIG.get("task"))
    dataset_path: str = CONFIG.get("dataset_path")
    output_dir: str = CONFIG.get("output_dir")
    device: str = CONFIG.get("device", "cpu")
    dataset_part: str = CONFIG.get("dataset_part")
    batch_size: int = int(CONFIG.get("batch_size"))

    if not dataset_path:
        raise Exception("Missing dataset_path in CONFIG")

    print(f"Loading dataset from: {dataset_path}")
    ds = load_from_disk(dataset_path)

    labels = _extract_labels(ds["labels"]) if "labels" in ds.column_names else None

    attr = AttributionMethod(
        method_name=method_name,
        task_index=task_index,
        label_index=0,
        model_dir=CONFIG.get("model_dir"),
        device=device,
    )

    _ensure_dir(output_dir)
    print(f"Writing xSMILES-compatible attributions to: {output_dir}")

    num_samples = len(ds["input_ids"])
    records = []

    input_ids_list = ds["input_ids"]
    attention_mask_list = ds["attention_mask"]
    smiles_list = ds["smiles"] if "smiles" in ds.column_names else None

    with tqdm(total=num_samples, desc="Attributing & writing", unit="samples") as pbar:
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)

            batch_ids_py, batch_mask_py = [], []
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
                skip_special_tokens=True,
            )

            with torch.no_grad():
                pred_batch = attr.wrapper(input_ids, attention_mask)
                pred_batch = pred_batch.detach().cpu().numpy().astype(float)

            for bi in range(end - start):
                ids = batch_ids_py[bi]
                mask = batch_mask_py[bi]

                seq_len = int(np.sum(mask))
                tokens = attr.tokenizer.convert_ids_to_tokens(ids[:seq_len])
                scores = attributions[bi, :seq_len].cpu().numpy().astype(float).tolist()

                # Filter out special tokens from tokens and scores
                special_tokens = set()
                if attr.tokenizer.pad_token is not None:
                    special_tokens.add(attr.tokenizer.pad_token)
                if attr.tokenizer.cls_token is not None:
                    special_tokens.add(attr.tokenizer.cls_token)
                if attr.tokenizer.sep_token is not None:
                    special_tokens.add(attr.tokenizer.sep_token)
                
                # Also check for special token IDs
                special_token_ids = set()
                if attr.tokenizer.pad_token_id is not None:
                    special_token_ids.add(attr.tokenizer.pad_token_id)
                if attr.tokenizer.cls_token_id is not None:
                    special_token_ids.add(attr.tokenizer.cls_token_id)
                if attr.tokenizer.sep_token_id is not None:
                    special_token_ids.add(attr.tokenizer.sep_token_id)
                
                # Filter tokens and scores, keeping only non-special tokens
                filtered_tokens = []
                filtered_scores = []
                for i, (token, token_id) in enumerate(zip(tokens, ids[:seq_len])):
                    if token not in special_tokens and token_id not in special_token_ids:
                        filtered_tokens.append(token)
                        filtered_scores.append(scores[i])
                
                tokens = filtered_tokens
                scores = filtered_scores

                smiles = smiles_list[start + bi] if smiles_list is not None else attr.tokenizer.decode(
                    [t for t, m in zip(ids, mask) if m == 1], skip_special_tokens=True
                )

                # Create character-level sequence and scores
                # Each character from a token gets the same score as that token
                char_sequence = []
                char_scores = []
                for token, score in zip(tokens, scores):
                    # Split token into units preserving two-character atoms like 'Cl' and 'Br'
                    token_chars = _split_smiles_preserving_atoms(token)
                    char_sequence.extend(token_chars)
                    # Duplicate the score for each character in the token
                    char_scores.extend([score] * len(token_chars))

                label_value = None
                if labels is not None:
                    try:
                        label_value = int(labels[start + bi])
                    except Exception:
                        label_value = float(labels[start + bi])

                record = {
                    "string": smiles,
                    "tokens": tokens,
                    "token_scores": scores,
                    "sequence": char_sequence,
                    "methods": [
                        {
                            "name": method_name,
                            "scores": char_scores,
                        }
                    ],
                    "attributes": {
                        "predicted_score": float(pred_batch[bi]),
                        "true_label": label_value,
                    }
                }
                records.append(record)

            pbar.update(end - start)

    single_out_path = os.path.join(output_dir, f"{dataset_part}.json")
    with open(single_out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Wrote xSMILES-compatible JSON: {single_out_path}")
