import argparse
from AttributionPipeline.config import CONFIG
from AttributionPipeline.src.attribute_dataset import run


def main():
    parser = argparse.ArgumentParser(description="Run attribution dataset pipeline with optional CONFIG overrides.")
    parser.add_argument("--model_dir", type=str, help="Path to model directory")
    parser.add_argument("--method_name", type=str, help="Attribution method name")
    parser.add_argument("--task", type=int, help="Task index")
    parser.add_argument("--dataset_part", type=str, choices=["train", "test"], help="Dataset split to process")
    parser.add_argument("--dataset_path", type=str, help="Path to dataset (HF load_from_disk)")
    parser.add_argument("--output_dir", type=str, help="Directory to write outputs")
    parser.add_argument("--device", type=str, help="Device to use (cpu, cuda, mps)")
    parser.add_argument("--batch_size", type=int, help="Batch size for attribution")

    args = parser.parse_args()

    overrides = {
        "model_dir": args.model_dir,
        "method_name": args.method_name,
        "task": args.task,
        "dataset_part": args.dataset_part,
        "dataset_path": args.dataset_path,
        "output_dir": args.output_dir,
        "device": args.device,
        "batch_size": args.batch_size,
    }
    for key, value in overrides.items():
        if value is not None:
            CONFIG[key] = value

    run()


if __name__ == "__main__":
    main()