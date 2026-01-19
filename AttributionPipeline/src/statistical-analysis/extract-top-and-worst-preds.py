#!/usr/bin/env python3
"""
Extract top 5 best and worst MAE predictions from attribution data.

This script processes all JSON files in the AttributionPipeline/data directory,
calculates MAE (Mean Absolute Error) for each sample, and extracts:
- Top 5 worst predictions (highest MAE)
- Top 5 best predictions (lowest MAE)

Output structure mirrors the input structure with additional metadata.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import argparse


def calculate_mae(predicted: float, true_label: float) -> float:
    """Calculate Mean Absolute Error for a single prediction."""
    return abs(predicted - true_label)


def normalize_scores(scores: List[float]) -> List[float]:
    """
    Normalize scores symmetrically to [-1, 1] range preserving zero.
    
    Formula: x / max(|min|, |max|)
    
    Args:
        scores: List of attribution scores
    
    Returns:
        List of normalized scores in [-1, 1] range
    """
    if not scores:
        return scores
    
    # Calculate scale as the maximum absolute value
    max_abs = max(abs(s) for s in scores)
    
    # Avoid division by zero
    if max_abs < 1e-9:
        return [0.0] * len(scores)
    
    # Symmetric normalization: preserve sign and zero point
    normalized = [s / max_abs for s in scores]
    return normalized


def process_json_file(
    input_path: Path,
    dataset_name: str,
    method_name: str,
    split_name: str,
    top_n: int
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Process a single JSON file and extract top 5 best and worst predictions.
    
    Args:
        input_path: Path to input JSON file
        dataset_name: Name of the dataset (e.g., 'bbbp', 'clintox')
        method_name: Attribution method name (e.g., 'shap', 'integrated_gradients')
        split_name: Dataset split (e.g., 'train', 'test')
    
    Returns:
        Dictionary with 'best' and 'worst' keys containing lists of samples
    """
    print(f"Processing: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Calculate MAE for each sample and add metadata
    samples_with_mae = []
    for sample in data:
        predicted = sample['attributes']['predicted_score']
        true_label = sample['attributes']['true_label']
        mae = calculate_mae(predicted, true_label)
        
        # Normalize token_scores to [-1, 1]
        if 'token_scores' in sample and sample['token_scores']:
            sample['token_scores'] = normalize_scores(sample['token_scores'])
        
        # Normalize method scores to [-1, 1]
        if 'methods' in sample:
            for method in sample['methods']:
                if 'scores' in method and method['scores']:
                    method['scores'] = normalize_scores(method['scores'])
        
        # Add metadata to attributes
        sample['attributes']['mae'] = mae
        sample['attributes']['dataset'] = dataset_name
        sample['attributes']['method'] = method_name
        sample['attributes']['split'] = split_name
        
        samples_with_mae.append(sample)
    
    # Sort by MAE
    sorted_by_mae = sorted(samples_with_mae, key=lambda x: x['attributes']['mae'])
    
    # Extract top N best (lowest MAE) and worst (highest MAE)
    best_n = sorted_by_mae[:top_n]
    worst_n = sorted_by_mae[-top_n:][::-1]  # Reverse to show highest MAE first
    
    print(f"  Total samples: {len(data)}")
    if best_n:
        print(f"  Best MAE: {best_n[0]['attributes']['mae']:.6f}")
    if worst_n:
        print(f"  Worst MAE: {worst_n[0]['attributes']['mae']:.6f}")
    
    return {
        'best': best_n,
        'worst': worst_n
    }



def replicate_structure_and_extract(input_dir: Path, output_dir: Path, top_n: int):
    """
    Replicate directory structure and extract top/worst predictions.
    
    Args:
        input_dir: Input data directory (e.g., AttributionPipeline/data)
        output_dir: Output directory for extracted predictions
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Find all JSON files (excluding the root inputxtest.json)
    json_files = []
    for json_path in input_dir.rglob('*.json'):
        # Skip files in the root directory
        if json_path.parent == input_dir:
            continue
        json_files.append(json_path)
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to process\n")
    
    # Process each JSON file
    for json_path in json_files:
        # Parse path structure: data/dataset/method/split.json
        relative_path = json_path.relative_to(input_dir)
        parts = relative_path.parts
        
        if len(parts) != 3:
            print(f"Skipping {json_path}: unexpected structure")
            continue
        
        dataset_name = parts[0]  # e.g., 'bbbp' or 'clintox'
        method_name = parts[1]   # e.g., 'shap', 'integrated_gradients'
        split_file = parts[2]     # e.g., 'train.json' or 'test.json'
        split_name = split_file.replace('.json', '')
        
        # Process the file
        results = process_json_file(
            json_path,
            dataset_name,
            method_name,
            split_name,
            top_n
        )
        
        # Create output directories
        output_subdir = output_dir / dataset_name / method_name
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        # Write best predictions
        best_output = output_subdir / f"{split_name}_best{top_n}.json"
        with open(best_output, 'w', encoding='utf-8') as f:
            json.dump(results['best'], f, ensure_ascii=False, indent=2)
        print(f"  ✓ Wrote: {best_output}")
        
        # Write worst predictions
        worst_output = output_subdir / f"{split_name}_worst{top_n}.json"
        with open(worst_output, 'w', encoding='utf-8') as f:
            json.dump(results['worst'], f, ensure_ascii=False, indent=2)
        print(f"  ✓ Wrote: {worst_output}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Extract top 5 best and worst MAE predictions from attribution data.'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='AttributionPipeline/data',
        help='Input data directory (default: AttributionPipeline/data)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='AttributionPipeline/src/statistical-analysis/data_top_worst',
        help='Output directory (default: AttributionPipeline/src/statistical-analysis/data_top_worst)'
    )
    
    parser.add_argument(
        '--top-n',
        type=int,
        default=5,
        help='Number of top/worst predictions to extract (default: 5)'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    top_n = args.top_n
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Extraction count (Top N): {top_n}")
    print("=" * 60)
    print()
    
    replicate_structure_and_extract(input_dir, output_dir, top_n)
    
    print("=" * 60)
    print("✓ Extraction complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
