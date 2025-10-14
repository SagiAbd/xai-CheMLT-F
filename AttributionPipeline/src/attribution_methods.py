import torch
import numpy as np
from typing import Optional, Tuple, List, Dict
from captum.attr import (
    LayerIntegratedGradients,
    LayerGradientXActivation,
    LayerDeepLift,
    LayerGradCam,
    LayerAttribution,
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


class AttributionMethod:
    """Unified interface for token-level attributions using layer-based Captum methods."""

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
        self.embeddings = self.model.encoder1.embeddings

        # Initialize the correct layer-based attribution method
        self.method = self._init_method()

    def _init_method(self):
        """Initialize layer-based Captum attribution method."""
        if self.method_name in ["integrated_gradients"]:
            return LayerIntegratedGradients(self.wrapper, self.embeddings)
        elif self.method_name in ["saliency", "input_x_gradient"]:
            # For these, use LayerGradientXActivation to handle embedding-level gradients
            return LayerGradientXActivation(self.wrapper, self.embeddings)
        elif self.method_name in ["deeplift", "layer_deeplift"]:
            return LayerDeepLift(self.wrapper, self.embeddings)
        elif self.method_name == "gradcam":
            # Optional example for completeness
            return LayerGradCam(self.wrapper, self.embeddings)
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
        baselines: Optional[torch.Tensor] = None,
        n_steps: int = 50,
        normalize: bool = True,
        return_convergence_delta: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute token-level attributions using layer-based methods."""
        self.model.eval()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Create baseline (used for IG and DeepLift)
        if baselines is None and self.method_name in ["integrated_gradients", "deeplift"]:
            baselines = torch.zeros_like(input_ids)
            baselines[:] = self.tokenizer.pad_token_id

        baseline_mask = torch.zeros_like(attention_mask)
        baselines = baselines.to(self.device)

        # Compute attributions
        delta = None
        if isinstance(self.method, LayerIntegratedGradients):
            attributions, delta = self.method.attribute(
                inputs=(input_ids, attention_mask),
                baselines=(baselines, baseline_mask),
                n_steps=n_steps,
                return_convergence_delta=True,
                **kwargs,
            )
        elif isinstance(self.method, LayerDeepLift):
            attributions = self.method.attribute(
                inputs=(input_ids, attention_mask),
                baselines=(baselines, baseline_mask),
                **kwargs,
            )
        else:
            # Gradient-based (LayerGradientXActivation, etc.)
            attributions = self.method.attribute(
                inputs=(input_ids, attention_mask),
                **kwargs,
            )

        # Sum across embedding dimension
        if attributions.dim() == 3:
            attributions = attributions.sum(dim=-1)

        # Apply mask and normalize
        if normalize:
            attributions = self._normalize(attributions, attention_mask)

        result_attrs = attributions.detach().cpu()
        if return_convergence_delta and delta is not None:
            return result_attrs, delta
        return result_attrs, None

    def _normalize(self, attributions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Normalize attributions per sample (L2 norm, masked)."""
        attributions = attributions * mask
        norm = torch.norm(attributions, dim=1, keepdim=True) + 1e-9
        return attributions / norm

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
