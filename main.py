import os
import json
import logging
import random
import numpy as np
from huggingface_hub import login
from src.preprocessing import preprocess_data, load_tokenizer, save_tokenized_data

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

def print_first_three(data: list):
    for i in range(0, 3):
        print(f"{data[i]}\n")
    print("\n")

def prepare_data(config): 
    # Get raw data
    train_data, val_data, test_data = preprocess_data(config=config)

    # Tokenize
    # Load tokenizer
    load_tokenizer()
    # Tokenize and save data
    save_tokenized_data(train_data, "data/train_tokenized.pt")
    save_tokenized_data(val_data, "data/val_tokenized.pt")
    save_tokenized_data(test_data, "data/test_tokenized.pt")

def call_pipeline(config):
    prepare_data(config=config)

    # dummy result
    return 0

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

    result = call_pipeline(config=config)

    # logging.info(f"Result: {result}")

    print("Run completed successfully.")

if __name__ == "__main__":
    main()