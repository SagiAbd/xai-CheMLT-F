#!/usr/bin/env python3
"""
Extract top 5 best/worst MAE predictions and heaviest/lightest molecules.

This script processes all JSON files in the AttributionPipeline/data directory,
calculates MAE (Mean Absolute Error) and Molecular Weight (using RDKit) for each sample,
and extracts for each label (0 and 1):
- Top N worst predictions (highest MAE)
- Top N best predictions (lowest MAE)
- Top N heaviest molecules (highest MW)
- Top N lightest molecules (lowest MW)

Output structure mirrors the input structure with additional metadata.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import argparse
from rdkit import Chem
from rdkit.Chem import Descriptors


def calculate_mae(predicted: float, true_label: float) -> float:
    """Calculate Mean Absolute Error for a single prediction."""
    return abs(predicted - true_label)

def calculate_mol_weight(smiles: str) -> float:
    """Calculate Molecular Weight using RDKit."""
    if not smiles:
        return 0.0
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return round(Descriptors.MolWt(mol), 2)
    except Exception as e:
        print(f"Warning: Failed to calculate MW for SMILES: {smiles}... Error: {e}")
    return 0.0

def normalize_scores(scores: List[float]) -> List[float]:
    """
    Normalize scores symmetrically to [-1, 1] range preserving zero.
    
    Formula: x / max(|min|, |max|)
    """
    if not scores:
        return scores
    
    max_abs = max(abs(s) for s in scores)
    
    if max_abs < 1e-9:
        return [0.0] * len(scores)
    
    normalized = [s / max_abs for s in scores]
    return normalized

def process_json_file(
    input_path: Path,
    dataset_name: str,
    method_name: str,
    split_name: str,
    top_n: int
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Process a single JSON file and extract top lists for each label.
    
    Returns:
        Dictionary keyed by label ('0' or '1'), each containing a dict of lists:
        {
            '0': {
                'best_mae': [...],
                'worst_mae': [...],
                'heaviest': [...],
                'lightest': [...]
            },
            '1': ...
        }
    """
    print(f"Processing: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process samples
    samples_by_label = {0: [], 1: []}
    
    for sample in data:
        predicted = sample['attributes']['predicted_score']
        true_label = sample['attributes']['true_label']
        mae = calculate_mae(predicted, true_label)
        
        smiles = sample.get('string', '')
        mw = calculate_mol_weight(smiles)
        
        # Normalize scores
        if 'token_scores' in sample and sample['token_scores']:
            sample['token_scores'] = normalize_scores(sample['token_scores'])
        
        if 'methods' in sample:
            for method in sample['methods']:
                if 'scores' in method and method['scores']:
                    method['scores'] = normalize_scores(method['scores'])
        
        # Add metadata
        sample['attributes']['mae'] = mae
        sample['attributes']['molecular_weight'] = mw
        sample['attributes']['dataset'] = dataset_name
        sample['attributes']['method'] = method_name
        sample['attributes']['split'] = split_name
        
        # Group by label
        label_int = int(true_label)
        if label_int in samples_by_label:
            samples_by_label[label_int].append(sample)
        else:
            # Fallback for unexpected labels, though usually binary 0/1
            if label_int not in samples_by_label:
                samples_by_label[label_int] = []
            samples_by_label[label_int].append(sample)
            
    results = {}
    
    for label, samples in samples_by_label.items():
        if not samples:
            continue
            
        # 1. Sort by MAE
        sorted_by_mae = sorted(samples, key=lambda x: x['attributes']['mae'])
        best_mae = sorted_by_mae[:top_n]
        worst_mae = sorted_by_mae[-top_n:][::-1] # Highest MAE first
        
        # 2. Sort by Molecular Weight
        sorted_by_mw = sorted(samples, key=lambda x: x['attributes']['molecular_weight'])
        lightest = sorted_by_mw[:top_n]
        heaviest = sorted_by_mw[-top_n:][::-1] # Heaviest first
        
        results[str(label)] = {
            'best_mae': best_mae,
            'worst_mae': worst_mae,
            'heaviest': heaviest,
            'lightest': lightest
        }
        
        print(f"  Label {label}: {len(samples)} samples")
    
    return results


def replicate_structure_and_extract(input_dir: Path, output_dir: Path, top_n: int):
    """
    Replicate directory structure and extract lists.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    json_files = []
    for json_path in input_dir.rglob('*.json'):
        if json_path.parent == input_dir:
            continue
        json_files.append(json_path)
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to process\n")
    
    for json_path in json_files:
        relative_path = json_path.relative_to(input_dir)
        parts = relative_path.parts
        
        if len(parts) != 3:
            print(f"Skipping {json_path}: unexpected structure")
            continue
        
        dataset_name = parts[0]
        method_name = parts[1]
        split_file = parts[2]
        split_name = split_file.replace('.json', '')
        
        results = process_json_file(
            json_path,
            dataset_name,
            method_name,
            split_name,
            top_n
        )
        
        # Create output directories
        output_subdir = output_dir / dataset_name / method_name / split_name
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        for label, categories in results.items():
            # suffix for filenames e.g., "label0"
            lbl_suffix = f"label{label}"
            
            # Write files
            files_map = {
                f"{lbl_suffix}_best_mae.json": categories['best_mae'],
                f"{lbl_suffix}_worst_mae.json": categories['worst_mae'],
                f"{lbl_suffix}_heaviest.json": categories['heaviest'],
                f"{lbl_suffix}_lightest.json": categories['lightest'],
            }
            
            for fname, content in files_map.items():
                out_path = output_subdir / fname
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(content, f, ensure_ascii=False, indent=2)
            
            print(f"  ✓ Wrote 4 files for label {label} in {output_subdir}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Extract top best/worst MAE and heaviest/lightest molecules.'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='AttributionPipeline/data',
        help='Input data directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='AttributionPipeline/src/statistical-analysis/data_top_worst',
        help='Output directory'
    )
    
    parser.add_argument(
        '--top-n',
        type=int,
        default=5,
        help='Number of samples to extract (default: 5)'
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
