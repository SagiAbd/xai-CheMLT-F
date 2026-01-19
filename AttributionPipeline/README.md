# AttributionPipeline

A pipeline for computing token-level and character-level attribution scores for molecular SMILES strings using the CheMLT-F multi-task model. This pipeline supports multiple explainability methods including gradient-based approaches (Integrated Gradients, Gradient X Activation, DeepLift, LRP) and SHAP.

## Overview

AttributionPipeline provides a unified interface for interpreting CheMLT-F model predictions on molecular property prediction tasks. It generates attribution scores that highlight which parts of a SMILES string contribute most to the model's predictions.

### Key Features

- **Multiple Attribution Methods**: Supports Captum-based methods (Integrated Gradients, Gradient X Activation, DeepLift, LRP) and SHAP
- **Multi-Task Support**: Works with all 11 tasks in the CheMLT-F model (8 classification, 3 regression)
- **Character-Level Attributions**: Preserves molecular semantics by keeping two-character atoms (Cl, Br, etc.) together
- **Batch Processing**: Efficiently processes entire datasets with progress tracking
- **xSMILES Compatible Output**: Generates JSON output compatible with xSMILES visualization tools

## Supported Tasks

The pipeline supports the following molecular property prediction tasks:

| Task ID | Dataset | Labels | Type |
|---------|---------|--------|------|
| 0 | BACE | 1 | Classification |
| 1 | HIV | 1 | Classification |
| 2 | BBBP | 1 | Classification |
| 3 | ClinTox | 2 | Classification |
| 4 | Tox21 | 12 | Classification |
| 5 | MUV | 17 | Classification |
| 6 | SIDER | 27 | Classification |
| 7 | ToxCast | 617 | Classification |
| 8 | Delaney | 1 | Regression |
| 9 | FreeSolv | 1 | Regression |
| 10 | Lipo | 1 | Regression |

## Project Structure

```
AttributionPipeline/
├── config.py                     # Configuration settings
├── src/
│   ├── __init__.py
│   ├── attribute_dataset.py      # Main dataset attribution script
│   ├── attribution_methods.py    # Attribution method implementations
│   └── utils.py                  # Model loading utilities
├── data/                         # Output directory for attribution results
│   ├── bbbp/
│   └── clintox/
└── deberta_xai_testing.ipynb     # Testing and visualization notebook
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- transformers
- captum
- shap
- datasets (HuggingFace)
- numpy
- tqdm

### Setup

```bash
# Install required packages
pip install torch transformers captum shap datasets numpy tqdm safetensors
```

## Configuration

Edit `config.py` to configure the attribution pipeline:

```python
CONFIG = {
    "model_dir": "Weights/Scaffold_CheMLT-F",        # Path to CheMLT-F model
    "method_name": "input_x_gradient",                # Attribution method
    "task": 3,                                        # Task ID (0-10)
    "dataset_part": "train",                          # Dataset split (train/test)
    "dataset_path": "Datasets/Scaffold_datasets/train_datasets/clintox",
    "output_dir": "AttributionPipeline/data/clintox/input_x_gradient",
    "device": "mps",                                  # Device: cpu/cuda/mps
    "batch_size": 1                                   # Batch size for processing
}
```

### Available Attribution Methods

- `integrated_gradients` - Layer-based Integrated Gradients
- `input_x_gradient` - Layer-based Gradient X Activation
- `deeplift` - Layer-based DeepLift
- `lrp` - Layer-based Layer-wise Relevance Propagation
- `shap` - SHAP values using token masking

## Usage

### Basic Usage

1. Configure the pipeline in `config.py`
2. Run the attribution script:

```python
from AttributionPipeline.src.attribute_dataset import run

run()
```

### Programmatic Usage

```python
from AttributionPipeline.src.attribution_methods import AttributionMethod

# Initialize attribution method
attr = AttributionMethod(
    method_name="integrated_gradients",
    task_index=3,              # ClinTox task
    label_index=1,             # For multi-label tasks
    model_dir="Weights/Scaffold_CheMLT-F",
    device="cuda"
)

# Get prediction and compute attributions
input_ids, attention_mask, prediction = attr.predict(
    "CCO",  # SMILES string
    max_length=512
)

attributions, convergence_delta = attr.compute(
    input_ids=input_ids,
    attention_mask=attention_mask,
    n_steps=50,
    normalize=True
)

# Get top contributing tokens
tokens = attr.decode_tokens(input_ids)
top_tokens = attr.get_top_tokens(tokens, attributions[0], top_k=5)
print(top_tokens)
```

### SHAP Visualization

```python
from AttributionPipeline.src.attribution_methods import AttributionMethod

attr = AttributionMethod(
    method_name="shap",
    task_index=2,  # BBBP
    model_dir="Weights/Scaffold_CheMLT-F"
)

# Generate SHAP visualization
attr.visualize_shap("CCO")
```

## Output Format

The pipeline generates JSON files in xSMILES-compatible format:

```json
[
  {
    "string": "CCO",
    "tokens": ["C", "C", "O"],
    "token_scores": [0.42, 0.35, 0.23],
    "sequence": ["C", "C", "O"],
    "methods": [
      {
        "name": "input_x_gradient",
        "scores": [0.42, 0.35, 0.23]
      }
    ],
    "attributes": {
      "predicted_score": 0.85,
      "true_label": 1.0
    }
  }
]
```

### Output Fields

- `string`: Original SMILES string
- `tokens`: Tokenized representation
- `token_scores`: Attribution scores at token level
- `sequence`: Character-level sequence (preserves two-character atoms)
- `methods`: List of attribution methods with character-level scores
- `attributes`: Prediction and ground truth label

## Features

### Atom-Preserving Tokenization

The pipeline intelligently splits SMILES strings while preserving two-character atoms (Cl, Br, He, Li, etc.) as single units, maintaining chemical semantics in the attribution scores.

### Multi-Label Support

For tasks with multiple labels (e.g., ClinTox with 2 labels), specify the `label_index` parameter to target specific predictions.

### Baseline Selection

Captum methods use a single baseline token ("C" - carbon) for computing attributions, providing chemically meaningful reference points.

### Normalization

Attribution scores can be normalized using L2 normalization across the sequence, useful for comparing attributions across different molecules.

## Examples

### Process Entire Dataset

```python
from AttributionPipeline.src.attribute_dataset import run
from AttributionPipeline.config import CONFIG

# Configure for ClinTox task
CONFIG["task"] = 3
CONFIG["dataset_path"] = "Datasets/Scaffold_datasets/train_datasets/clintox"
CONFIG["output_dir"] = "AttributionPipeline/data/clintox/integrated_gradients"
CONFIG["method_name"] = "integrated_gradients"

# Run pipeline
run()
```

### Compare Multiple Methods

```python
from AttributionPipeline.src.attribution_methods import AttributionMethod

methods = ["integrated_gradients", "input_x_gradient", "shap"]
smiles = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"

for method_name in methods:
    attr = AttributionMethod(
        method_name=method_name,
        task_index=2,  # BBBP
        model_dir="Weights/Scaffold_CheMLT-F"
    )
    
    input_ids, attention_mask, pred = attr.predict(smiles)
    attributions, _ = attr.compute(input_ids, attention_mask)
    
    tokens = attr.decode_tokens(input_ids)
    top_tokens = attr.get_top_tokens(tokens, attributions[0], top_k=3)
    
    print(f"\n{method_name}:")
    print(f"Prediction: {pred:.4f}")
    print(f"Top tokens: {top_tokens}")
```

## Implementation Details

### Model Architecture

The pipeline uses the CheMLT-F model, a DeBERTa-based multi-task architecture fine-tuned on molecular property prediction. The model processes SMILES strings and outputs predictions for 11 different molecular property tasks.

### Attribution Methods

#### Captum Methods
- Use layer-based approaches targeting the embedding layer
- Compute attributions via gradients and layer activations
- Support convergence delta calculation for Integrated Gradients

#### SHAP
- Uses token masking approach
- Computes Shapley values for each token
- Provides force plot visualizations

### Special Token Handling

The pipeline automatically filters out special tokens ([CLS], [SEP], [PAD]) from attribution outputs, focusing only on the meaningful SMILES content.

## Contributing

When extending the pipeline:

1. Add new attribution methods to `AttributionMethod` class in `attribution_methods.py`
2. Update `CONFIG` in `config.py` for new parameters
3. Maintain xSMILES output format compatibility
4. Preserve atom-level tokenization logic for chemical correctness

## Citation

If you use this pipeline in your research, please cite the CheMLT-F model and relevant attribution method papers.

## License

Please refer to the parent project license.
