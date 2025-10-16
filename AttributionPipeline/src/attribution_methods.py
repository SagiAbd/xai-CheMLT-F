import torch
import numpy as np
import shap
from typing import Optional, Tuple, List, Dict
from captum.attr import (
    LayerIntegratedGradients,
    LayerGradientXActivation,
    LayerDeepLift
)
from AttributionPipeline.src.utils import load_model
from AttributionPipeline.config import CONFIG, TASKS


class CheMLTWrapper(torch.nn.Module):
    """Wrapper for CheMLT model to work with Captum layer-based attribution methods."""

    def __init__(self, model, task_index: int, label_index: int = 0):
        """
        Args:
            model: The CheMLT multitask model
            task_index: Which task to interpret (0-10)
            label_index: For multi-label tasks, which label to interpret
        """
        super().__init__()
        self.model = model
        self.task_index = task_index
        self.label_index = label_index
        self.task_name, self.num_labels, self.task_type = TASKS[task_index]

    def forward(self, input_ids, attention_mask):
        """Forward pass returning scalar prediction for the target label."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_ids2=None,
            attention_mask2=None,
            input_ids3=None,
            attention_mask3=None,
            task_index=self.task_index,
        )

        logits = outputs["logits"]
        if self.task_type == "classification":
            probs = torch.sigmoid(logits)
            if self.num_labels == 1:
                return probs.squeeze(-1)
            else:
                return probs[:, self.label_index]
        else:
            return logits.squeeze(-1)


class CheMLTPipeline:
    """Pipeline wrapper for SHAP compatibility."""
    
    def __init__(self, model, tokenizer, task_index: int, label_index: int = 0, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.task_index = task_index
        self.label_index = label_index
        self.device = device
        self.wrapper = CheMLTWrapper(model, task_index, label_index)
        self.task_name, self.num_labels, self.task_type = TASKS[task_index]
        
    def __call__(self, smiles_list: List[str]):
        """
        SHAP-compatible call method.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            List of predictions (probabilities or regression values)
        """
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
            
        results = []
        for smiles in smiles_list:
            inputs = self.tokenizer(
                smiles,
                return_tensors="pt",
                max_length=512,
                padding="max_length",
                truncation=True,
            )
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            with torch.no_grad():
                pred = self.wrapper(input_ids, attention_mask).cpu().item()
            
            results.append(pred)
            
        return np.array(results)


class AttributionMethod:
    """Unified interface for token-level attributions using layer-based Captum methods and SHAP."""

    def __init__(
        self,
        method_name: str,
        task_index: int,
        label_index: int = 0,
        model_dir: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        model_path = model_dir or CONFIG.get("model_dir")
        self.model, self.tokenizer = load_model(model_path, device)
        self.model.eval()

        self.method_name = method_name.lower()
        self.device = device
        self.task_index = task_index
        self.label_index = label_index
        self.task_name, self.num_labels, self.task_type = TASKS[task_index]

        self.wrapper = CheMLTWrapper(self.model, task_index, label_index)
        
        if self.method_name != "shap":
            self.embeddings = self.model.encoder1.embeddings
            self.method = self._init_method()
        else:
            # Initialize SHAP-specific components
            self.pipeline = CheMLTPipeline(
                self.model, self.tokenizer, task_index, label_index, device
            )
            self.explainer = None  # Initialized lazily in compute()

    def _init_method(self):
        """Initialize layer-based Captum attribution method."""
        if self.method_name in ["integrated_gradients"]:
            return LayerIntegratedGradients(self.wrapper, self.embeddings)
        elif self.method_name in ["input_x_gradient"]:
            return LayerGradientXActivation(self.wrapper, self.embeddings)
        elif self.method_name in ["deeplift"]:
            return LayerDeepLift(self.wrapper, self.embeddings)
        else:
            raise ValueError(f"Unknown or unsupported attribution method: {self.method_name}")

    def predict(self, smiles: str, max_length: int = 512) -> Dict[str, any]:
        """Tokenize SMILES and get model prediction."""
        inputs = self.tokenizer(
            smiles,
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        with torch.no_grad():
            pred_value = self.wrapper(input_ids, attention_mask).item()

        if self.task_type == "classification":
            pred_label = (
                f"Active (prob={pred_value:.3f})"
                if pred_value > 0.5
                else f"Inactive (prob={pred_value:.3f})"
            )
        else:
            pred_label = f"Value={pred_value:.3f}"

        return {
            "pred_value": pred_value,
            "pred_label": pred_label,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "tokens": tokens,
        }

    def compute(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        baseline_smiles_list: Optional[List[str]] = None,
        n_steps: int = 50,
        normalize: bool = True,
        return_convergence_delta: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute token-level attributions.
        
        For SHAP: Uses the SHAP Explainer on the raw SMILES string
        For Captum methods: Uses ensemble of baselines
        """
        if self.method_name == "shap":
            return self._compute_shap(input_ids, attention_mask, **kwargs)
        else:
            return self._compute_captum(
                input_ids, attention_mask, baseline_smiles_list, 
                n_steps, normalize, return_convergence_delta, **kwargs
            )

    def _compute_shap(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, None]:
        """Compute SHAP values for token-level attributions."""
        # Decode input_ids back to SMILES
        smiles = self.tokenizer.decode(
            input_ids[0], 
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False
        )
        
        # Initialize explainer if not already done
        if self.explainer is None:
            self.explainer = shap.Explainer(self.pipeline, self.tokenizer)
        
        # Compute SHAP values
        shap_values = self.explainer([smiles])
        
        # Extract token-level attributions
        # shap_values.values has shape (1, num_tokens, num_outputs)
        # For single output, we take [:, :, 0]
        attributions = torch.tensor(shap_values.values[0, :])
        
        # Pad or truncate to match input_ids length
        seq_len = input_ids.shape[1]
        if attributions.shape[0] < seq_len:
            # Pad with zeros
            padding = torch.zeros(seq_len - attributions.shape[0])
            attributions = torch.cat([attributions, padding])
        elif attributions.shape[0] > seq_len:
            # Truncate
            attributions = attributions[:seq_len]
        
        # Add batch dimension
        attributions = attributions.unsqueeze(0)
        
        return attributions, None

    def _compute_captum(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        baseline_smiles_list: Optional[List[str]] = None,
        n_steps: int = 50,
        normalize: bool = True,
        return_convergence_delta: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute token-level attributions using Captum methods with ensemble of baselines.
        """
        self.model.eval()
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        if baseline_smiles_list is None:
            baseline_smiles_list = ["C"]  # simple neutral molecule

        ensemble_attributions = []
        deltas = []

        for ref_smiles in baseline_smiles_list:
            ref_inputs = self.tokenizer(
                ref_smiles,
                return_tensors="pt",
                max_length=input_ids.shape[1],
                padding="max_length",
                truncation=True,
            )
            baseline_ids = ref_inputs["input_ids"].to(self.device)

            # Preserve special tokens like CLS and SEP 
            special_tokens = {
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id,
            }
            # copy special tokens from input_ids to baseline_ids
            for i in range(baseline_ids.shape[1]):
                if input_ids[0, i].item() in special_tokens:
                    baseline_ids[0, i] = input_ids[0, i]

            baseline_mask = torch.zeros_like(attention_mask)

            # Compute single baseline attribution
            if isinstance(self.method, LayerIntegratedGradients):
                attrs, delta = self.method.attribute(
                    inputs=(input_ids, attention_mask),
                    baselines=(baseline_ids, baseline_mask),
                    n_steps=n_steps,
                    return_convergence_delta=True,
                    **kwargs,
                )
                deltas.append(delta)
            elif isinstance(self.method, LayerDeepLift):
                attrs = self.method.attribute(
                    inputs=(input_ids, attention_mask),
                    baselines=(baseline_ids, baseline_mask),
                    **kwargs,
                )
            else:
                attrs = self.method.attribute(
                    inputs=(input_ids, attention_mask),
                    **kwargs,
                )

            if attrs.dim() == 3:
                attrs = attrs.sum(dim=-1)

            ensemble_attributions.append(attrs.detach())

        # Average over ensemble baselines
        attributions = torch.mean(torch.stack(ensemble_attributions), dim=0)

        # Optional normalization
        if normalize:
            attributions = self._normalize(attributions, attention_mask)

        delta_mean = torch.mean(torch.stack(deltas), dim=0) if deltas else None

        attributions = attributions.detach().cpu()
        
        return attributions, delta_mean

    def _normalize(self, attributions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Normalize attributions per sample (L2 norm, masked)."""
        attributions = attributions * mask
        norm = torch.norm(attributions, dim=1, keepdim=True) + 1e-9
        return attributions / norm

    def visualize_shap(self, smiles: str):
        """
        Create SHAP text visualization for a SMILES string.
        
        Args:
            smiles: Input SMILES string
        """
        if self.method_name != "shap":
            raise ValueError("visualize_shap() can only be called when method_name='shap'")
        
        if self.explainer is None:
            self.explainer = shap.Explainer(self.pipeline)
        
        # Get prediction
        prediction = self.pipeline([smiles])
        print(f"Prediction: {prediction[0]:.4f}")
        
        # Compute and visualize SHAP values
        shap_values = self.explainer([smiles])
        shap.plots.text(shap_values)
        
        return shap_values

    def get_top_tokens(
        self,
        tokens: List[str],
        attributions: np.ndarray,
        top_k: int = 5,
        include_special: bool = False,
    ) -> List[Tuple[str, float]]:
        """Return the most important tokens."""
        if not include_special:
            valid_indices = [
                i
                for i, token in enumerate(tokens)
                if token
                not in [
                    self.tokenizer.pad_token,
                    self.tokenizer.cls_token,
                    self.tokenizer.sep_token,
                ]
            ]
        else:
            valid_indices = [
                i for i, token in enumerate(tokens) if token != self.tokenizer.pad_token
            ]

        valid_tokens = [tokens[i] for i in valid_indices]
        valid_attrs = attributions[valid_indices]
        sorted_indices = np.argsort(np.abs(valid_attrs))[::-1]

        return [
            (valid_tokens[i], float(valid_attrs[i]))
            for i in sorted_indices[:top_k]
        ]

    def decode_tokens(self, input_ids: torch.Tensor) -> List[List[str]]:
        """Decode token IDs into readable tokens."""
        return [
            self.tokenizer.convert_ids_to_tokens(seq.tolist()) for seq in input_ids
        ]


# Example usage
if __name__ == "__main__":
    # Example with SHAP
    attributor = AttributionMethod(
        method_name="shap",
        task_index=0,
        label_index=0,
        device="cuda"
    )
    
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    
    # Get prediction
    pred_info = attributor.predict(smiles)
    print(f"Prediction: {pred_info['pred_label']}")
    
    # Compute attributions
    attributions, _ = attributor.compute(
        pred_info["input_ids"],
        pred_info["attention_mask"]
    )
    
    # Get top contributing tokens
    top_tokens = attributor.get_top_tokens(
        pred_info["tokens"],
        attributions[0].numpy(),
        top_k=5
    )
    print("\nTop contributing tokens:")
    for token, score in top_tokens:
        print(f"  {token}: {score:.4f}")
    
    # Visualize with SHAP
    attributor.visualize_shap(smiles)