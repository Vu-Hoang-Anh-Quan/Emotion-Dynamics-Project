import os
from dotenv import load_dotenv
import json
import logging
from pathlib import Path
import random
import torch
import numpy as np
from huggingface_hub import login
from src.preprocessing.preprocess import preprocess_and_save_data
from load_config import load_config, apply_cli_overrides, apply_overrides
from src.paths import ProjectPaths
from src.pipelines.utterance_pipeline import run_utterance_pipeline
from src.pipelines.conversation_pipeline import run_conversation_pipeline

paths: ProjectPaths
HUGGING_FACE_KEY: str

def load_env():
    global HUGGING_FACE_KEY, paths
    load_dotenv(dotenv_path=paths.root / ".env")
    HUGGING_FACE_KEY = os.getenv("HUGGING_FACE_KEY")

def log_config(config: dict): # Just print out the current using config
    formatted = json.dumps(config, indent=2, sort_keys=True, default=str)
    for line in formatted.splitlines():
        logging.info(line)

def setup_experiment(config):
    global paths
    exp_dir = paths.experiments / config["experiment_name"]

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

def dummy_return():
    print("Return earlier than usual")
    logging.info("Return earlier than usual, has completed the run")
    return 0, 0, 0, 0

def main():
    # Path
    global paths
    paths = ProjectPaths(Path(__file__).parent)

    # Colab compatibility will be added later, for now just run on local with config that is set to cpu or cuda based on availability

    # 1. Load config in regard of cuda availability
    config = load_config(paths / "configs" / f'default_{"cuda" if torch.cuda.is_available() else "cpu"}.json')
    manual_overrides = {
                            "experiment_name": "Custom pooling v1 - Mean pooling",
                            # "prepare_data_again": 1,
                            "deterministic_run": 0, 
                            # "compile_model": 1,
                            # "debug": 1,
                            "bert.freeze_except_last_k": 8,
                            # "head.lr": 5e-4,
                            # "attention.dropout": 0.2,
                            "attention.dim": 256,
                            # "use_amp": 0,
                        }
    config = apply_overrides(config, manual_overrides)
    config = apply_cli_overrides(config)

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

    # Prep data
    if config["prepare_data_again"]: preprocess_and_save_data(config, paths)

    # Utterance pipeline
    if config["utterance_recognition"]["run"]:
        print("Running utterance pipeline...")
        test_loss, test_accuracy, test_f1_score_macro, test_f1_m_ex = run_utterance_pipeline(config, paths)
        print(f"Utterance Pipeline - Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.4f} | Test F1-score macro: {test_f1_score_macro:.4f} | Test F1-score macro non-Neutral: {test_f1_m_ex:.4f}")
        logging.info(f"Utterance Pipeline - Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.4f} | Test F1-score macro: {test_f1_score_macro:.4f} | Test F1-score macro non-Neutral: {test_f1_m_ex:.4f}")
    # Conversation pipeline
    if config["conversation_recognition"]["run"]:
        print("Running conversation pipeline...")
        test_loss, test_accuracy, test_f1_score_macro, test_f1_m_ex = run_conversation_pipeline(config, paths)
        print(f"Conversation Pipeline - Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.4f} | Test F1-score macro: {test_f1_score_macro:.4f} | Test F1-score macro non-Neutral: {test_f1_m_ex:.4f}")
        logging.info(f"Conversation Pipeline - Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.4f} | Test F1-score macro: {test_f1_score_macro:.4f} | Test F1-score macro non-Neutral: {test_f1_m_ex:.4f}")

    print("Run completed successfully.")

if __name__ == "__main__":
    main()