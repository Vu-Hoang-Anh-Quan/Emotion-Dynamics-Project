import os
import json
import logging
import random
import numpy as np
from huggingface_hub import login
from src.preprocessing import preprocess_data

def load_config(path="configs/default.json"):
    with open(path, "r") as f:
        return json.load(f)

def setup_experiment(config):
    exp_dir = os.path.join("experiments", config["experiment_name"])
    os.makedirs(exp_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(exp_dir, "log.txt"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    return exp_dir

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def dummy_pipeline():
    # Placeholder for future pipeline
    data = ["hello", "how are you"]
    processed = [x.upper() for x in data]
    result = len(processed)
    return result

def main():
    # 1. Load config
    config = load_config()

    # login to huggingface
    login(config["huggingface_token"])

    # 2. Setup experiment
    exp_dir = setup_experiment(config)

    # 3. Set seed
    set_seed(config["seed"])

    # logging.info(f"Starting experiment: {config['experiment_name']}")

    # 4. Run pipeline

    train_data, val_data, test_data = preprocess_data(config=config)
    for i in range(0, 3):
        print(f"{train_data[i]}\n")
    print("\n")
    for i in range(0, 3):
        print(f"{val_data[i]}\n")
    print("\n")
    for i in range(0, 3):
        print(f"{test_data[i]}\n")
    print("\n")

    result = dummy_pipeline()

    # logging.info(f"Result: {result}")

    print("Run completed successfully.")

if __name__ == "__main__":
    main()