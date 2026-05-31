import os
from dotenv import load_dotenv
import json
import logging
from pathlib import Path
import random
import torch
import numpy as np
import torch.nn as nn
from huggingface_hub import login
from src.preprocessing import preprocess_data, save_data
from src.dataloader.dataloader import build_dataloaders, load_tokenizer
from src.models.bert_classifier import BertClassifier
from src.training.trainer import train_model, get_final_test_accuracy, load_model

project_root: Path
data_root: Path
HUGGING_FACE_KEY: str

def load_env():
    global HUGGING_FACE_KEY, project_root
    load_dotenv(dotenv_path=project_root / ".env")
    HUGGING_FACE_KEY = os.getenv("HUGGING_FACE_KEY")

def load_config(path, overrides = {}):
    with open(path, "r") as f:
        config = json.load(f)
    config.update(overrides)
    return config

def log_config(config: dict): # Just print out the current using config
    formatted = json.dumps(config, indent=2, sort_keys=True, default=str)
    for line in formatted.splitlines():
        logging.info(line)

def setup_experiment(config):
    global data_root
    exp_dir = data_root / "experiments" / str(config['experiment_name'])
    exp_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=exp_dir / "log.txt",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    return exp_dir

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)


def print_first_three(data: list):
    for i in range(0, 3):
        print(f"{data[i]}\n")
    print("\n")

def debug_dataloader(dataloader):
    batch = next(iter(dataloader))
    print(batch["input_ids"].shape)      # [B, T, L]
    print_first_three(batch["input_ids"])
    print(batch["attention_mask"].shape)
    print_first_three(batch["attention_mask"])
    print(batch["labels"].shape)         # [B, T]
    print_first_three(batch["labels"])

def debug_nan(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(name)

def dummy_return():
    print("Return earlier than usual")
    logging.info("Return earlier than usual, has completed the run")
    return 0, 0, 0, 0

def prepare_data(config):
    global data_root
    # Get raw data
    train_data, val_data, test_data = preprocess_data(config=config)

    # print_first_three(train_data)
    
    # Save data, as tokenzing happens later
    data_dir = data_root / "data"
    data_dir.mkdir(exist_ok=True)
    save_data(train_data, data_dir / "train_tokenized.pt")
    save_data(val_data, data_dir / "val_tokenized.pt")
    save_data(test_data, data_dir / "test_tokenized.pt")

def call_pipeline(config):
    global data_root

    if config["prepare_data_again"]: prepare_data(config=config)

    # dummy return
    # return dummy_return()

    # Ensure that the path exists
    model_dir = data_root / "saved_models"
    model_dir.mkdir(exist_ok=True)

    # Setup model path and device
    MODEL_PATH = model_dir / f"{config['resulting_model_name']}.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    load_tokenizer() # Put here so that you actually logged in before

    # Load data
    data_dir = data_root / "data"
    test_loader = build_dataloaders(
        data = torch.load(data_dir / "test_tokenized.pt" ,map_location=device),
        batch_size=config["batch_size"],
        do_shuffling=False
    )

    print(f"Test data succesfully loaded from {data_dir / 'test_tokenized.pt'}")
    logging.info(f"Test data succesfully loaded from {data_dir / 'test_tokenized.pt'}")

    # debug_dataloader(test_loader)
    # return dummy_return()

    # Build Model
    model = BertClassifier(
        model_name=config["embedding_model_name"],
        num_labels=config["num_labels"],
        dropout_bert=config["dropout_bert"],
        dropout_head=config["dropout_head"],
        freeze_except_last_k=config["freeze_except_last_k"],
        max_turns=config['max_turns'],
        dropout_attention=config['dropout_attention'],
    ).to(device) # Load the model to cuda/cpu

    print(f"Model {config['resulting_model_name']} successfully built")
    logging.info(f"Model {config['resulting_model_name']} successfully built")

    if (not(os.path.exists(MODEL_PATH)) or config["need_to_retrain"]):
        # If the model not existed yet or said to retrain in config
        print("Training model from scratch...")

        train_loader = build_dataloaders(
            data = torch.load(data_dir / "train_tokenized.pt",map_location=device),
            batch_size=config["batch_size"],
            do_shuffling=True
        )
        val_loader = build_dataloaders(
            data = torch.load(data_dir / "val_tokenized.pt",map_location=device),
            batch_size=config["batch_size"],
            do_shuffling=False
        )
        print(f"Train and Val data succesfully loaded from {data_dir}")
        logging.info(f"Train and Val data succesfully loaded from {data_dir}")

        train_model(model, train_loader, val_loader, config, model_path=MODEL_PATH)

    # Load the best model
    print(f"Loading model {MODEL_PATH}")
    load_model(model, MODEL_PATH, config["compile_model"])
    print("Model loaded succesfully")

    # Final test with test_data
    test_loss, test_accuracy, test_f1_m, test_f1_m_ex = get_final_test_accuracy(model, test_loader, device)

    # Return test_loss and test_accurcacy
    return test_loss, test_accuracy, test_f1_m, test_f1_m_ex

def main():
    global project_root, data_root, HUGGING_FACE_KEY

    project_root = Path(__file__).resolve().parent
    data_root = project_root
    # Check if in Colab
    try:
        from google.colab import drive # type: ignore
        # drive.mount('/content/drive')
        data_root = Path("/content/drive/MyDrive/Emotional Dynamics Project/")
        # Put your base path here to your project
    except ImportError:
        pass

    # 1. Load config in regard of cuda availability
    config = load_config(project_root / "configs" / f'default_{"cuda" if torch.cuda.is_available() else "cpu"}.json',
                         {
                            "experiment_name": "Custom pooling v1 - Mean pooling",
                            # "prepare_data_again": 1,
                            # "need_to_retrain": 1,
                            "epochs": 8,
                            "deterministic_run": 0, 
                            # "compile_model": 1,
                            # "debug": 1,
                            "batch_size": 4,
                            "freeze_except_last_k": 8,
                            # "lr_head": 5e-4,
                            # "dropout_attention": 0.2,
                            "max_turns": 36,
                            "attention_dim": 256,
                            # "use_amp": 0,
                            "resulting_model_name": "Custom pooling v1"
                         }
                         )

    # Load env
    load_env()

    # Check if requires deterministic run
    if (config["deterministic_run"]):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 2. Setup experiment
    exp_dir = setup_experiment(config)

    # login to huggingface
    print(HUGGING_FACE_KEY)
    login(HUGGING_FACE_KEY)

    # 3. Set seed
    set_seed(config["seed"])

    # Add logging about your training loss and val loss, val acc 
    logging.info(f"Starting experiment: {config['experiment_name']}")

    # Add logging for my config for each run
    log_config(config)

    # 4. Run pipeline

    test_loss, test_accuracy, test_f1_score_macro, test_f1_m_ex = call_pipeline(config=config)

    print(f"Final test loss: {test_loss:.4f}\nFinal test accuracy: {test_accuracy:.4f}\nFinal F1-score macro: {test_f1_score_macro:.4f}\nFinal F1-score macro non-Neutral: {test_f1_m_ex:.4f}")

    logging.info(f"Final test loss: {test_loss:.4f}\nFinal test accuracy: {test_accuracy:.4f}\nFinal F1-score macro: {test_f1_score_macro:.4f}\nFinal F1-score macro non-Neutral: {test_f1_m_ex:.4f}")

    print("Run completed successfully.")

if __name__ == "__main__":
    main()