# test_model.py
"""Simple test script for CheMLT-F SMILES prediction."""

import torch
from transformers import AutoTokenizer
from Models.chemlt_f_model import DebertaMultiTaskModel

# Task definitions
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


def load_model(model_dir):
    """Load CheMLT-F model from HuggingFace format directory."""
    
    print(f"Loading model from {model_dir}...")
    
    # Initialize model architecture
    model = DebertaMultiTaskModel(
        model_path1=model_dir,  # Load config from here
        model_path2=model_dir,  # Same path (SMILES-only)
        num_labels_list = [1, 1, 1, 2, 12, 17, 27, 617, 1, 1, 1],
        problem_type_list = [
            "classification", "classification", "classification",
            "classification", "classification", "classification",
            "classification", "classification",
            "regression", "regression", "regression"
        ]
    )
    
    # Load weights from safetensors or pytorch_model.bin
    try:
        from safetensors.torch import load_file
        state_dict = load_file(f"{model_dir}/model.safetensors")
        print("✅ Loaded from model.safetensors")
    except:
        state_dict = torch.load(f"{model_dir}/pytorch_model.bin", map_location="cpu")
        print("✅ Loaded from pytorch_model.bin")
    
    # Filter out encoder2 (for SMILES-only)
    filtered = {k: v for k, v in state_dict.items() if not k.startswith("encoder2.")}
    model.load_state_dict(filtered, strict=False)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    print("✅ Tokenizer loaded\n")
    
    return model, tokenizer


def predict(model, tokenizer, smiles, task_index):
    """
    Run prediction for a single SMILES string on a specific task.

    Args:
        model: The multitask DeBERTa model.
        tokenizer: The corresponding tokenizer.
        smiles (str): The SMILES string to predict on.
        task_index (int): The index of the task (based on TASKS list).

    Returns:
        dict: A structured result with probabilities or values.
    """
    model.eval()

    # Tokenize the SMILES string
    inputs = tokenizer(
        smiles,
        return_tensors="pt",
        max_length=512,
        padding="max_length",
        truncation=True
    )

    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            input_ids2=None,
            attention_mask2=None,
            input_ids3=None,
            attention_mask3=None,
            task_index=task_index
        )

    logits = outputs["logits"]
    task_name, num_labels, task_type = TASKS[task_index]

    # --- Handle classification ---
    if task_type == "classification":
        probs = torch.sigmoid(logits).squeeze()

        if num_labels == 1:
            # Binary classification → scalar probability
            prob = probs.item()
            prediction = "Active" if prob > 0.5 else "Inactive"
            return {
                "task": task_name,
                "num_labels": num_labels,
                "probability": prob,
                "prediction": prediction
            }

        else:
            # Multi-label classification → vector of probabilities
            probs = probs.tolist()
            return {
                "task": task_name,
                "num_labels": num_labels,
                "probabilities": probs
            }

    # --- Handle regression ---
    elif task_type == "regression":
        value = logits.squeeze().item()
        return {
            "task": task_name,
            "num_labels": num_labels,
            "value": value
        }

    # --- Unexpected type ---
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    

def main():
    # Test molecules
    test_molecules = {
        "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",  # ❌ Low BBB permeability (polar acidic)
        "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(O)=O",  # ❌ Low – mostly peripheral
        "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # ✅ High – CNS stimulant
        "Paracetamol": "CC(=O)Nc1ccc(O)cc1",  # ✅ Moderate–High – centrally acting
        "Penicillin": "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O",  # ❌ Very low – polar, ionized
        "Diazepam": "CN1C(=O)CN=C(c2ccccc2)c3cc(Cl)ccc13",  # ✅ High – benzodiazepine, CNS active
        "Lidocaine": "CCN(CC)C(=O)C1=CC=CC=C1",  # ✅ High – anesthetic, crosses BBB
        "Vancomycin": "CC1=C(C(=O)NC2=CC=CC=C2)C(=O)NC3=C(C=CC(=C3O)O)O1",  # ❌ Very low – large polar antibiotic
    }
    
    # Load model (just point to the directory!)
    model, tokenizer = load_model("Weights/Scaffold_CheMLT-F")
    
    # Test on different tasks
    test_tasks = [
        (2, "BBBP - Blood-Brain Barrier Permeability"),
        (10, "Lipo - Lipophilicity (logD)"),
        (4, "Tox21 - 12 Toxicity Pathways"),
    ]
    
    for task_idx, task_desc in test_tasks:
        print("=" * 80)
        print(f"{task_desc}")
        print("=" * 80)
        
        for drug_name, smiles in test_molecules.items():
            result = predict(model, tokenizer, smiles, task_idx)
            
            # Format output
            if "probability" in result:
                print(f"{drug_name:12} | {result['prediction']:8} | Prob: {result['probability']:.3f}")
            elif "value" in result:
                print(f"{drug_name:12} | Value: {result['value']:.3f}")
            else:
                # Multi-label - show first 5 probabilities
                probs = result['probabilities'][:5]
                print(f"{drug_name:12} | First 5: {[f'{p:.2f}' for p in probs]}")
        
        print()


if __name__ == "__main__":
    main()