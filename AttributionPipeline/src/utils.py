import torch
from Models.chemlt_f_model import DebertaMultiTaskModel
from transformers import AutoTokenizer
from datasets import load_from_disk
import numpy as np


def load_model(model_dir, device):
    """Load CheMLT-F model from HuggingFace format directory."""
    
    print(f"Loading model from {model_dir}...")
    
    # Initialize model architecture
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
    except:
        state_dict = torch.load(f"{model_dir}/pytorch_model.bin", map_location="cpu")
        print("✅ Loaded from pytorch_model.bin")
    
    # Filter out encoder2 (for SMILES-only)
    filtered = {k: v for k, v in state_dict.items() if not k.startswith("encoder2.")}
    model.load_state_dict(filtered, strict=False)
    model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    print("✅ Tokenizer loaded\n")
    
    return model, tokenizer
