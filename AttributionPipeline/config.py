CONFIG = {
    "model_dir": "Weights/Scaffold_CheMLT-F",
    "method_name": "shap",
    "task": 2,
    "dataset_part": "test", # test/train
    # Absolute or relative path to the dataset directory (HuggingFace load_from_disk export)
    "dataset_path": "Datasets/Scaffold_datasets/test_datasets/bbbp",
    # Where to write attribution JSON files
    "output_dir": "AttributionPipeline/data/bbbp/shap",
    "device": "cuda",
    "batch_size": 1
}

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
