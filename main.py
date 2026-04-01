import os
import json
import logging
import random
import torch
import numpy as np
import torch.nn as nn
from huggingface_hub import login
from src.preprocessing import preprocess_data, load_tokenizer, save_tokenized_data
from src.data.dataloader import build_dataloaders
from src.models.bert_classifier import BertClassifier
from src.training.trainer import train_model, get_final_test_accuracy

def load_config(path="configs/default_cpu.json"):
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
    # prepare_data(config=config)
    # Turn off the above line if u want to prep ur data again

    # Setup model path and device
    MODEL_PATH = f"saved_models/{config['resulting_model_name']}/.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader, test_loader = build_dataloaders( # Exactly in this order
        train_tokenized = torch.load("data/train_tokenized.pt", device),
        val_tokenized = torch.load("data/val_tokenized.pt", device),
        test_tokenized = torch.load("data/test_tokenized.pt", device),
        batch_size = config['batch_size']
    )

    # Build Model
    model = BertClassifier(
        model_name=config["embedding_model_name"],
        num_labels=config["num_labels"],
        dropout=config["dropout_rate"]
    ).to(device) # Load the model to the device cuda/cpu

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
        train_model(model, train_loader, val_loader, config)
        torch.save(model.state_dict(), MODEL_PATH) # Just save the current, not the best in the training process
    else:
        print(f"Loading model {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH))

    # Final test with test_data
    test_loss, test_accuracy = get_final_test_accuracy(model, test_loader, device)

    print(f"Final test loss: {test_loss}\nFinal test accuracy: {test_accuracy}")

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