# %% [markdown]
# # Interpreting CheMLT-F SMILES Predictions with Captum
# 
# This notebook demonstrates how to use Captum to interpret predictions from the CheMLT-F model,
# helping understand which tokens in a SMILES string contribute most to molecular property predictions.

# %% [markdown]
# ## 1. Setup and Imports

# %%
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from captum.attr import LayerIntegratedGradients, TokenReferenceBase
from captum.attr import visualization as viz

# Import your model
from Models.chemlt_f_model import DebertaMultiTaskModel

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## 2. Task Definitions

# %%
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

# %% [markdown]
# ## 3. Load Model and Tokenizer

# %%
def load_model(model_dir):
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

# Load the model
MODEL_DIR = "Weights/Scaffold_CheMLT-F"
model, tokenizer = load_model(MODEL_DIR)

# %% [markdown]
# ## 4. Create Model Wrapper for Captum
# 
# Captum requires a forward function that takes token IDs and returns predictions.
# We'll create a wrapper that handles this for our multi-task model.

# %%
class CheMLTWrapper(torch.nn.Module):
    """Wrapper for CheMLT model to work with Captum."""
    
    def __init__(self, model, task_index, label_index=0):
        """
        Args:
            model: The CheMLT multitask model
            task_index: Which task to interpret (0-10)
            label_index: For multi-label tasks, which label to interpret (default: 0)
        """
        super().__init__()
        self.model = model
        self.task_index = task_index
        self.label_index = label_index
        self.task_name, self.num_labels, self.task_type = TASKS[task_index]
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass returning scalar prediction for the target label.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_ids2=None,
            attention_mask2=None,
            input_ids3=None,
            attention_mask3=None,
            task_index=self.task_index
        )
        
        logits = outputs["logits"]
        
        # For classification, return probability
        if self.task_type == "classification":
            probs = torch.sigmoid(logits)
            if self.num_labels == 1:
                return probs.squeeze(-1)
            else:
                return probs[:, self.label_index]
        # For regression, return raw value
        else:
            return logits.squeeze(-1)

# %% [markdown]
# ## 5. Prediction Function

# %%
def predict_smiles(model, tokenizer, smiles, task_index, label_index=0):
    """
    Get prediction and confidence for a SMILES string.
    
    Returns:
        pred_value: The prediction (probability or regression value)
        pred_label: Human-readable prediction
        token_ids: Token IDs for interpretation
        tokens: Actual tokens
    """
    # Tokenize
    inputs = tokenizer(
        smiles,
        return_tensors="pt",
        max_length=512,
        padding="max_length",
        truncation=True
    )
    
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Create wrapper
    wrapper = CheMLTWrapper(model, task_index, label_index)
    
    # Predict
    with torch.no_grad():
        pred_value = wrapper(input_ids, attention_mask).item()
    
    # Format prediction
    task_name, num_labels, task_type = TASKS[task_index]
    
    if task_type == "classification":
        pred_label = f"Active (prob={pred_value:.3f})" if pred_value > 0.5 else f"Inactive (prob={pred_value:.3f})"
    else:
        pred_label = f"Value={pred_value:.3f}"
    
    return pred_value, pred_label, input_ids, attention_mask, tokens

# %% [markdown]
# ## 6. Attribution Methods
# 
# We'll use Layer Integrated Gradients, which attributes predictions to input tokens
# by computing gradients along the path from a baseline to the input.

# %%
def compute_attributions(model, input_ids, attention_mask, task_index, label_index=0):
    """
    Compute token attributions using Layer Integrated Gradients.
    
    Returns:
        attributions: Attribution scores for each token
        delta: Approximation error
    """
    # Create wrapper
    wrapper = CheMLTWrapper(model, task_index, label_index)
    
    # Get embedding layer
    embeddings = model.encoder1.embeddings
    
    # Create Layer Integrated Gradients
    lig = LayerIntegratedGradients(wrapper, embeddings)
    
    # Create baseline (PAD token)
    baseline_ids = torch.zeros_like(input_ids).long()
    baseline_ids[:] = tokenizer.pad_token_id
    baseline_mask = torch.zeros_like(attention_mask)
    
    # Compute attributions
    attributions, delta = lig.attribute(
        inputs=(input_ids, attention_mask),
        baselines=(baseline_ids, baseline_mask),
        return_convergence_delta=True,
        n_steps=50
    )
    
    # Sum across embedding dimension
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.cpu().detach().numpy()
    
    return attributions, delta

# %% [markdown]
# ## 7. Visualization Functions

# %%
def visualize_attributions(tokens, attributions, pred_label, smiles):
    """
    Create HTML visualization of token attributions.
    """
    # Filter out padding tokens
    valid_indices = [i for i, token in enumerate(tokens) if token != tokenizer.pad_token]
    tokens = [tokens[i] for i in valid_indices]
    attributions = attributions[valid_indices]
    
    # Normalize attributions for visualization
    attr_sum = attributions.sum()
    normalized_attrs = attributions / attr_sum if attr_sum != 0 else attributions
    
    # Create visualization records
    vis_records = []
    for token, attr in zip(tokens, normalized_attrs):
        vis_records.append(
            viz.VisualizationDataRecord(
                word_attributions=attr,
                pred_prob=0,  # Not used in our visualization
                pred_class=pred_label,
                true_class="",
                attr_class=token,
                attr_score=attr,
                raw_input_ids="",
                convergence_score=0
            )
        )
    
    # Display
    print(f"\n{'='*80}")
    print(f"SMILES: {smiles}")
    print(f"Prediction: {pred_label}")
    print(f"{'='*80}\n")
    
    html = viz.visualize_text(vis_records)
    return html

# %% [markdown]
# ## 8. Example Interpretations
# 
# Let's interpret predictions for several molecules across different tasks.

# %% [markdown]
# ### Example 1: Blood-Brain Barrier Permeability (BBBP)

# %%
# Test molecules
test_molecules = {
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Expected: High BBB permeability
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",  # Expected: Low BBB permeability
    "Diazepam": "CN1C(=O)CN=C(c2ccccc2)c3cc(Cl)ccc13",  # Expected: High BBB permeability
}

task_idx = 2  # BBBP task
task_name = TASKS[task_idx][0]

print(f"\n{'#'*80}")
print(f"# Task: {task_name} - Blood-Brain Barrier Permeability")
print(f"{'#'*80}\n")

for drug_name, smiles in test_molecules.items():
    print(f"\n--- Analyzing: {drug_name} ---")
    
    # Get prediction
    pred_value, pred_label, input_ids, attention_mask, tokens = predict_smiles(
        model, tokenizer, smiles, task_idx
    )
    
    # Compute attributions
    attributions, delta = compute_attributions(
        model, input_ids, attention_mask, task_idx
    )
    
    # Visualize
    html = visualize_attributions(tokens, attributions, pred_label, smiles)
    
    # Show top contributing tokens
    valid_indices = [i for i, token in enumerate(tokens) if token != tokenizer.pad_token]
    valid_tokens = [tokens[i] for i in valid_indices]
    valid_attrs = attributions[valid_indices]
    
    # Sort by absolute attribution
    sorted_indices = np.argsort(np.abs(valid_attrs))[::-1]
    
    print("\nTop 5 Contributing Tokens:")
    for i in sorted_indices[:5]:
        token = valid_tokens[i]
        attr = valid_attrs[i]
        direction = "→ Active" if attr > 0 else "→ Inactive"
        print(f"  {token:10s} | Attribution: {attr:+.4f} {direction}")
    
    print(f"\nConvergence Delta: {delta.item():.6f}")

# %% [markdown]
# ### Example 2: Lipophilicity (Lipo)

# %%
task_idx = 10  # Lipo task
task_name = TASKS[task_idx][0]

print(f"\n{'#'*80}")
print(f"# Task: {task_name} - Lipophilicity Prediction")
print(f"{'#'*80}\n")

test_molecules_lipo = {
    "Ethanol": "CCO",  # Low lipophilicity
    "Hexane": "CCCCCC",  # High lipophilicity
    "Phenol": "Oc1ccccc1",  # Moderate lipophilicity
}

for drug_name, smiles in test_molecules_lipo.items():
    print(f"\n--- Analyzing: {drug_name} ---")
    
    pred_value, pred_label, input_ids, attention_mask, tokens = predict_smiles(
        model, tokenizer, smiles, task_idx
    )
    
    attributions, delta = compute_attributions(
        model, input_ids, attention_mask, task_idx
    )
    
    html = visualize_attributions(tokens, attributions, pred_label, smiles)
    
    # Show top contributing tokens
    valid_indices = [i for i, token in enumerate(tokens) if token != tokenizer.pad_token]
    valid_tokens = [tokens[i] for i in valid_indices]
    valid_attrs = attributions[valid_indices]
    sorted_indices = np.argsort(np.abs(valid_attrs))[::-1]
    
    print("\nTop 5 Contributing Tokens:")
    for i in sorted_indices[:5]:
        token = valid_tokens[i]
        attr = valid_attrs[i]
        direction = "→ Higher" if attr > 0 else "→ Lower"
        print(f"  {token:10s} | Attribution: {attr:+.4f} {direction}")

# %% [markdown]
# ### Example 3: Multi-label Toxicity (Tox21)
# 
# For multi-label tasks, we can interpret each label separately.

# %%
task_idx = 4  # Tox21 (12 labels)
task_name = TASKS[task_idx][0]

# Pick a test molecule
smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # Caffeine
drug_name = "Caffeine"

print(f"\n{'#'*80}")
print(f"# Task: {task_name} - Toxicity Pathways (Multi-label)")
print(f"# Molecule: {drug_name}")
print(f"{'#'*80}\n")

# Interpret first 3 toxicity labels
for label_idx in range(3):
    print(f"\n--- Label {label_idx} ---")
    
    pred_value, pred_label, input_ids, attention_mask, tokens = predict_smiles(
        model, tokenizer, smiles, task_idx, label_idx=label_idx
    )
    
    attributions, delta = compute_attributions(
        model, input_ids, attention_mask, task_idx, label_idx=label_idx
    )
    
    print(f"Prediction: {pred_label}")
    
    # Show top contributing tokens
    valid_indices = [i for i, token in enumerate(tokens) if token != tokenizer.pad_token]
    valid_tokens = [tokens[i] for i in valid_indices]
    valid_attrs = attributions[valid_indices]
    sorted_indices = np.argsort(np.abs(valid_attrs))[::-1]
    
    print("Top 3 Contributing Tokens:")
    for i in sorted_indices[:3]:
        token = valid_tokens[i]
        attr = valid_attrs[i]
        print(f"  {token:10s} | Attribution: {attr:+.4f}")

# %% [markdown]
# ## 9. Interactive Analysis Function
# 
# Create a reusable function for quick interpretation of any SMILES.

# %%
def interpret_smiles(smiles, task_index, label_index=0, top_k=5):
    """
    Complete interpretation pipeline for a SMILES string.
    
    Args:
        smiles: SMILES string to interpret
        task_index: Task index (0-10)
        label_index: For multi-label tasks, which label to interpret
        top_k: Number of top contributing tokens to display
    """
    task_name, num_labels, task_type = TASKS[task_index]
    
    print(f"\n{'='*80}")
    print(f"Task: {task_name} | SMILES: {smiles}")
    print(f"{'='*80}\n")
    
    # Predict
    pred_value, pred_label, input_ids, attention_mask, tokens = predict_smiles(
        model, tokenizer, smiles, task_index, label_index
    )
    
    print(f"Prediction: {pred_label}\n")
    
    # Compute attributions
    attributions, delta = compute_attributions(
        model, input_ids, attention_mask, task_index, label_index
    )
    
    # Show top tokens
    valid_indices = [i for i, token in enumerate(tokens) if token != tokenizer.pad_token]
    valid_tokens = [tokens[i] for i in valid_indices]
    valid_attrs = attributions[valid_indices]
    sorted_indices = np.argsort(np.abs(valid_attrs))[::-1]
    
    print(f"Top {top_k} Contributing Tokens:")
    for i in sorted_indices[:top_k]:
        token = valid_tokens[i]
        attr = valid_attrs[i]
        print(f"  {token:10s} | {attr:+.4f}")
    
    print(f"\nConvergence Delta: {delta.item():.6f}")
    
    # Create visualization
    html = visualize_attributions(tokens, attributions, pred_label, smiles)
    
    return pred_value, attributions, html

# %% [markdown]
# ## 10. Try Your Own SMILES!

# %%
# Example usage - customize these
your_smiles = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"  # Ibuprofen
your_task = 2  # BBBP

result = interpret_smiles(your_smiles, your_task, top_k=10)

# %% [markdown]
# ## Summary
# 
# This notebook demonstrates how to:
# 
# 1. Load and wrap the CheMLT-F model for Captum
# 2. Compute token-level attributions using Layer Integrated Gradients
# 3. Visualize which SMILES tokens contribute most to predictions
# 4. Interpret both binary and multi-label classification tasks
# 5. Analyze regression predictions (lipophilicity, solubility, etc.)
# 
# Key insights:
# - **Functional groups** often show high attribution (e.g., -OH, -COOH, rings)
# - **Polarity markers** influence BBB permeability predictions
# - **Aromatic rings** typically contribute to lipophilicity
# - **Attribution magnitudes** indicate feature importance
# - **Sign of attribution** shows direction of influence (positive/negative)