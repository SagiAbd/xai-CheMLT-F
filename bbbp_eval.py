# bbbp_eval.py
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer
from Models.chemlt_f_model import DebertaMultiTaskModel

# === TASK DEFINITION ===
TASKS = {
    0: ("BACE", 1, "classification"),
    1: ("HIV", 1, "classification"),
    2: ("BBBP", 1, "classification"),
    3: ("ClinTox", 2, "classification"),
    4: ("Tox21", 12, "classification"),
    5: ("MUV", 17, "classification"),
    6: ("SIDER", 27, "classification"),
    7: ("ToxCast", 617, "classification"),
    8: ("Delaney", 1, "regression"),
    9: ("FreeSolv", 1, "regression"),
    10: ("Lipo", 1, "regression"),
}


# === LOAD MODEL ===
def load_model(model_dir):
    print(f"Loading model from {model_dir}...")

    model = DebertaMultiTaskModel(
        model_path1=model_dir,
        model_path2=model_dir,
        num_labels_list=[1, 1, 1, 2, 12, 17, 27, 617, 1, 1, 1],
        problem_type_list=[
            "classification", "classification", "classification",
            "classification", "classification", "classification",
            "classification", "classification",
            "regression", "regression", "regression"
        ]
    )

    # Load weights
    try:
        from safetensors.torch import load_file
        state_dict = load_file(f"{model_dir}/model.safetensors")
        print("✅ Loaded from model.safetensors")
    except Exception:
        state_dict = torch.load(f"{model_dir}/pytorch_model.bin", map_location="cpu")
        print("✅ Loaded from pytorch_model.bin")

    filtered = {k: v for k, v in state_dict.items() if not k.startswith("encoder2.")}
    model.load_state_dict(filtered, strict=False)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    print("✅ Tokenizer loaded\n")

    return model, tokenizer


# === EVALUATION ===
def evaluate_bbbp(model, dataset_path, task_index=2):
    print(f"📂 Loading pre-tokenized BBBP dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)

    # Some exports have nested splits; handle both
    if isinstance(dataset, dict) or hasattr(dataset, "keys"):
        ds = dataset.get("test", next(iter(dataset.values())))
    else:
        ds = dataset

    input_ids = ds["input_ids"]
    attention_mask = ds["attention_mask"]
    labels = np.array([x[0] if isinstance(x, list) else x for x in ds["labels"]]).astype(float)

    print(f"✅ Loaded {len(labels)} samples")

    preds = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(input_ids)), desc="Predicting"):
            # Convert to regular Python lists if necessary
            ids = input_ids[i]
            mask = attention_mask[i]

            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            if hasattr(mask, "tolist"):
                mask = mask.tolist()

            inputs = {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.tensor([mask], dtype=torch.long),
                "input_ids2": None,
                "attention_mask2": None,
                "input_ids3": None,
                "attention_mask3": None,
                "task_index": task_index
            }

            outputs = model(**inputs)
            prob = torch.sigmoid(outputs["logits"]).item()
            preds.append(prob)

    auc = roc_auc_score(labels, preds)
    print(f"\n=== BBBP Evaluation ===")
    print(f"ROC-AUC: {auc:.4f}")
    print("=======================\n")

    return auc


# === MAIN ===
def main():
    model_dir = "Weights/Scaffold_CheMLT-F"
    dataset_path = "Datasets/Scaffold_datasets/test_datasets/bbbp"
    model, tokenizer = load_model(model_dir)

    auc = evaluate_bbbp(model, dataset_path, task_index=2)
    print(f"Final ROC-AUC on BBBP test set: {auc:.4f}")


if __name__ == "__main__":
    main()
