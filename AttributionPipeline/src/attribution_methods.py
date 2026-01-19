import torch
import numpy as np
import shap
from typing import Optional, Tuple, List, Dict, Any
from captum.attr import (
    LayerIntegratedGradients,
    LayerGradientXActivation,
    LayerDeepLift,
    LayerLRP
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
    
    def __init__(self, model, tokenizer, task_index: int, label_index: int = 0, device: str = "cpu"):
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
        Returns probabilities for both classes if classification, else regression values.
        """
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]

        preds = []
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

            preds.append(pred)

        preds = np.array(preds)

        # Convert scalar outputs to 2D probabilities if classification
        if self.task_type == "classification":
            # Ensure preds are between 0 and 1 (they should be sigmoid outputs)
            preds = np.clip(preds, 0, 1)
            preds = np.stack([1 - preds, preds], axis=1)  # shape (n, 2)
        
        # If regression, just return shape (n, 1)
        else:
            preds = preds.reshape(-1, 1)

        return preds


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
        
        if self.method_name == "shap":
            # SHAP setup
            self.pipeline = CheMLTPipeline(
                self.model, self.tokenizer, task_index, label_index, device
            )
            self.explainer = None
        else:
            # Captum setup
            self.embeddings = self.model.encoder1.embeddings
            self.method = self._init_method()

    def _init_method(self):
        """Initialize layer-based Captum attribution method."""
        if self.method_name in ["integrated_gradients"]:
            return LayerIntegratedGradients(self.wrapper, self.embeddings)
        elif self.method_name in ["input_x_gradient"]:
            return LayerGradientXActivation(self.wrapper, self.embeddings)
        elif self.method_name in ["deeplift"]:
            return LayerDeepLift(self.wrapper, self.embeddings)
        elif self.method_name in ["lrp"]:
            self.embeddings = self.model.encoder1.encoder.layer[-1]
            return LayerLRP(self.wrapper, self.embeddings)
        else:
            raise ValueError(f"Unknown or unsupported attribution method: {self.method_name}")

    def predict(self, smiles: str, max_length: int = 512) -> Dict[str, Any]:
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
        n_steps: int = 50,
        return_convergence_delta: bool = False,
        skip_special_tokens: bool = True,
        **kwargs,
    ):
        if self.method_name == "shap":
            return self._compute_shap(input_ids, attention_mask, skip_special_tokens=skip_special_tokens, **kwargs)
        else:
            return self._compute_captum(
                input_ids,
                attention_mask,
                n_steps,
                return_convergence_delta,
                skip_special_tokens=skip_special_tokens,
                **kwargs,
            )

    def _compute_shap(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        skip_special_tokens: bool,
        **kwargs
    ) -> Tuple[torch.Tensor, None]:
        """Compute SHAP values for token-level attributions."""
        # Decode input_ids back to SMILES
        smiles = self.tokenizer.decode(
            input_ids[0], 
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False
        )
        
        # Initialize explainer if not already done
        if self.explainer is None:
            self.explainer = shap.Explainer(self.pipeline, self.tokenizer)
        
        # Compute SHAP values
        shap_values = self.explainer([smiles])
        
        # Extract token-level attributions for the correct output
        if self.task_type == "classification":
            # For binary classification, attributions for positive class (index 1)
            attributions = torch.tensor(shap_values.values[0, :, 1], dtype=torch.float32)
        else:
            # For regression, single output (index 0)
            attributions = torch.tensor(shap_values.values[0, :, 0], dtype=torch.float32)
        
        # Pad or truncate to match input_ids length
        seq_len = input_ids.shape[1]
        if attributions.shape[0] < seq_len:
            padding = torch.zeros(seq_len - attributions.shape[0], dtype=torch.float32)
            attributions = torch.cat([attributions, padding])
        elif attributions.shape[0] > seq_len:
            attributions = attributions[:seq_len]
        
        # Add batch dimension and move to correct device
        attributions = attributions.unsqueeze(0).to(self.device)
        
        return attributions, None

    def _compute_captum(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        n_steps: int = 50,
        return_convergence_delta: bool = False,
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute token-level attributions using Captum methods with a single baseline ("C").
        """
        self.model.eval()
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Use a single simple baseline
        ref_smiles = "C"
        ref_inputs = self.tokenizer(
            ref_smiles,
            return_tensors="pt",
            max_length=input_ids.shape[1],
            padding="max_length",
            truncation=True,
        )
        baseline_ids = ref_inputs["input_ids"].to(self.device)

        if not skip_special_tokens:
            # Preserve special tokens like CLS and SEP
            special_tokens = {
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id,
            }
            for i in range(baseline_ids.shape[1]):
                if input_ids[0, i].item() in special_tokens:
                    baseline_ids[0, i] = input_ids[0, i]

        # Create proper baseline mask (attend to non-padding tokens)
        baseline_mask = (baseline_ids != self.tokenizer.pad_token_id).long().to(self.device)

        # Compute attributions
        if isinstance(self.method, LayerIntegratedGradients):
            attrs, delta = self.method.attribute(
                inputs=(input_ids, attention_mask),
                baselines=(baseline_ids, baseline_mask),
                n_steps=n_steps,
                return_convergence_delta=True,
                **kwargs,
            )
        elif isinstance(self.method, LayerDeepLift):
            attrs = self.method.attribute(
                inputs=(input_ids, attention_mask),
                baselines=(baseline_ids, baseline_mask),
                **kwargs,
            )
            delta = None
        else:
            attrs = self.method.attribute(
                inputs=(input_ids, attention_mask),
                **kwargs,
            )
            delta = None

        # If the attribution has embedding dimension, sum over it
        if attrs.dim() == 3:
            attrs = attrs.sum(dim=-1)

        attrs = attrs.detach().cpu()
        delta = delta.detach().cpu() if delta is not None else None

        return attrs, delta



    def visualize_shap(self, smiles: str):
        """
        Create SHAP text visualization for a SMILES string.
        
        Args:
            smiles: Input SMILES string
        """
        if self.method_name != "shap":
            raise ValueError("visualize_shap() can only be called when method_name='shap'")
        
        if self.explainer is None:
            self.explainer = shap.Explainer(self.pipeline, self.tokenizer)
        
        # Get prediction
        prediction = self.pipeline([smiles])
        print(f"Prediction: {prediction[0]}")
        
        # Compute and visualize SHAP values
        shap_values = self.explainer([smiles])
        shap.plots.text(shap_values)
        
        return shap_values

    def get_top_tokens(self, tokens, attributions, top_k=5, include_special=False):
        """Get top-k tokens by attribution magnitude."""
        attributions = np.array(attributions).flatten()

        if not include_special:
            valid_indices = [
                i for i, token in enumerate(tokens)
                if token not in [
                    self.tokenizer.pad_token,
                    self.tokenizer.cls_token,
                    self.tokenizer.sep_token,
                ]
            ]
        else:
            valid_indices = [
                i for i, token in enumerate(tokens)
                if token != self.tokenizer.pad_token
            ]

        valid_tokens = [tokens[i] for i in valid_indices]
        valid_attrs = attributions[valid_indices]

        sorted_indices = np.argsort(np.abs(valid_attrs))[::-1]
        top_tokens = [(valid_tokens[i], float(valid_attrs[i])) for i in sorted_indices[:top_k]]

        return top_tokens

    def decode_tokens(self, input_ids: torch.Tensor) -> List[List[str]]:
        """Decode token IDs into readable tokens."""
        return [
            self.tokenizer.convert_ids_to_tokens(seq.tolist()) for seq in input_ids
        ]
        