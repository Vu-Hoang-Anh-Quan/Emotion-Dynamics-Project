import os
from dotenv import load_dotenv
import json
import logging
import random
import torch
import numpy as np
import torch.nn as nn
from huggingface_hub import login
from src.preprocessing import preprocess_data, load_tokenizer, save_tokenized_data
from src.dataloader.dataloader import build_dataloaders
from src.models.bert_classifier import BertClassifier
from src.training.trainer import train_model, get_final_test_accuracy, load_model

base_path: str
HUGGING_FACE_KEY: str

def load_env():
    global HUGGING_FACE_KEY, base_path
    load_dotenv(dotenv_path=f"{base_path}.env")
    HUGGING_FACE_KEY = os.getenv("HUGGING_FACE_KEY")

def load_config(path="configs/default_cpu.json", overrides = {}):
    with open(path, "r") as f:
        config = json.load(f)
    config.update(overrides)
    return config

def log_config(config: dict):
    formatted = json.dumps(config, indent=2, sort_keys=True, default=str)
    for line in formatted.splitlines():
        logging.info(line)

def setup_experiment(config):
    global base_path
    exp_dir = f"{base_path}experiments/{config['experiment_name']}"
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
    global base_path
    # Get raw data
    train_data, val_data, test_data = preprocess_data(config=config)

    # Tokenize
    # Load tokenizer
    load_tokenizer()
    # Tokenize and save data
    save_tokenized_data(train_data, f"{base_path}data/train_tokenized.pt")
    save_tokenized_data(val_data, f"{base_path}data/val_tokenized.pt")
    save_tokenized_data(test_data, f"{base_path}data/test_tokenized.pt")

def call_pipeline(config):
    global base_path

    if config["prepare_data_again"]: prepare_data(config=config)

    # Ensure that the path exists
    os.makedirs(f"{base_path}saved_models", exist_ok=True)

    # Setup model path and device
    MODEL_PATH = f"{base_path}saved_models/{config['resulting_model_name']}.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    test_loader = build_dataloaders(
        tokenized_data = torch.load(f"{base_path}data/test_tokenized.pt",map_location=device),
        batch_size=config["batch_size"],
        do_shuffling=False
    )

    print(f"Test data succesfully loaded from {base_path}data")
    logging.info(f"Test data succesfully loaded from {base_path}data")

    # Build Model
    model = BertClassifier(
        model_name=config["embedding_model_name"],
        num_labels=config["num_labels"],
        dropout=config["dropout_rate"]
    ).to(device) # Load the model to cuda/cpu

    print(f"Model {config['resulting_model_name']} built successfully")
    logging.info(f"Model {config['resulting_model_name']} built successfully")

    # batch = next(iter(train_loader))
    # logits = model(batch["input_ids"], batch["attention_mask"])
    # print(logits.shape)
    # loss_function = nn.CrossEntropyLoss()
    # loss = loss_function(logits, batch["labels"])
    # print(loss.item())

    # return 0 # Test model forward pass

    if (not(os.path.exists(MODEL_PATH)) or config["need_to_retrain"]):
        # If the model not existed yet or said to retrain in config
        print("Training model from scratch...")

        train_loader = build_dataloaders(
            tokenized_data = torch.load(f"{base_path}data/train_tokenized.pt",map_location=device),
            batch_size=config["batch_size"],
            do_shuffling=True
        )
        val_loader = build_dataloaders(
            tokenized_data = torch.load(f"{base_path}data/val_tokenized.pt",map_location=device),
            batch_size=config["batch_size"],
            do_shuffling=False
        )
        print(f"Train and Val data succesfully loaded from {base_path}data")
        logging.info(f"Train and Val data succesfully loaded from {base_path}data")

        train_model(model, train_loader, val_loader, config, model_path=MODEL_PATH)
    else:
        print(f"Loading model {MODEL_PATH}")
        load_model(model, MODEL_PATH)

    # Final test with test_data
    test_loss, test_accuracy, test_f1_m = get_final_test_accuracy(model, test_loader, device)

    # Return test_loss and test_accurcacy
    return test_loss, test_accuracy, test_f1_m

def main():
    global base_path, HUGGING_FACE_KEY
    # Check if in Colab
    try:
        from google.colab import drive # type: ignore
        # drive.mount('/content/drive')
        base_path = "/content/drive/MyDrive/Emotional Dynamics Project/"
        # Put your base path here to your project
    except ImportError:
        base_path = ""

    # 1. Load config in regard of cuda availability
    config = load_config(f'configs/default_{"cuda" if torch.cuda.is_available() else "cpu"}.json',
                         {
                             "experiment_name": "Testing"
                         }
                         )

    # Load env
    load_env()

    # login to huggingface
    print(HUGGING_FACE_KEY)
    login(HUGGING_FACE_KEY)
    # Remember to invalidate and refresh the token again to be used

    # 2. Setup experiment
    exp_dir = setup_experiment(config)

    # 3. Set seed
    set_seed(config["seed"])

    # Add logging about your training loss and val loss, val acc 
    logging.info(f"Starting experiment: {config['experiment_name']}")

    # Add logging for my config for each run
    log_config(config)

    # 4. Run pipeline

    test_loss, test_accuracy, test_f1_score_macro = call_pipeline(config=config)

    print(f"Final test loss: {test_loss:.4f}\nFinal test accuracy: {test_accuracy:.4f}\nFinal F1-score macro: {test_f1_score_macro:.4f}")

    logging.info(f"Final test loss: {test_loss:.4f}\nFinal test accuracy: {test_accuracy:.4f}\nFinal F1-score macro: {test_f1_score_macro:.4f}")

    print("Run completed successfully.")

if __name__ == "__main__":
    main()