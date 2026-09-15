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
from src.utils.load_config import load_config, apply_cli_overrides, apply_overrides
from src.utils.paths import ProjectPaths
from src.pipelines.utterance_pipeline import run_utterance_pipeline
from src.pipelines.conversation_pipeline import run_conversation_pipeline
from src.utils.logging_utils import setup_logging

paths: ProjectPaths
HUGGING_FACE_KEY: str

def load_env():
    global HUGGING_FACE_KEY, paths
    load_dotenv(dotenv_path=paths.root / ".env")
    HUGGING_FACE_KEY = os.getenv("HUGGING_FACE_KEY")

def log_config(config: dict, logger: logging.Logger): # Just print out the current using config
    formatted = json.dumps(config, indent=2, sort_keys=True, default=str)
    for line in formatted.splitlines():
        logger.info(line)

def setup_experiment(config):
    global paths
    exp_dir = paths.experiments / config["experiment_name"]
    setup_logging(exp_dir)

def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

def dummy_return():
    print("Return earlier than usual")
    logging.info("Return earlier than usual, has completed the run")
    return 0, 0, 0, 0

def main():
    # Path
    global paths
    paths = ProjectPaths(Path(__file__).parent)
    paths.ensure_init_directories()

    # Load config
    config = load_config(paths / "configs" / f'final_experiment.json')
    manual_overrides = {
                            # "seed": 7,
                            # "prepare_data_again": 0,
                            # "utterance_recognition.run": False,
                            # "conversation_recognition.retrain": False
                        }
    config = apply_overrides(config, manual_overrides)
    config = apply_cli_overrides(config)

    # Load env
    load_env()

    # Setup experiment and logging
    setup_experiment(config)
    logger = logging.getLogger("main")
    logger.info("Experiment setup complete.")

    # login to huggingface
    print(HUGGING_FACE_KEY)
    login(HUGGING_FACE_KEY)

    # Set seed
    set_seed(config["seed"], config["deterministic_run"])

    # Add logging about your training loss and val loss, val acc 
    logger.info(f"Starting experiment: {config['experiment_name']}")

    # Add logging for my config for each run
    log_config(config, logger)

    # Prep data
    if config["prepare_data_again"]: preprocess_and_save_data(config, paths)

    # Utterance pipeline
    if config["utterance_recognition"]["run"]:
        logger.info("\n\nRunning utterance pipeline...")
        test_loss, test_accuracy, test_f1_score_macro, test_f1_m_ex = run_utterance_pipeline(config, paths)
        logger.info(f"Utterance Pipeline:\nTest Loss: {test_loss:.4f}\nTest Acc: {test_accuracy:.4f}\nTest F1-score macro: {test_f1_score_macro:.4f}\nTest F1-score macro non-Neutral: {test_f1_m_ex:.4f}")
    # Conversation pipeline
    if config["conversation_recognition"]["run"]:
        logger.info("\n\nRunning conversation pipeline...")
        test_loss, test_accuracy, test_f1_score_macro, test_f1_m_ex = run_conversation_pipeline(config, paths)
        logger.info(f"Conversation Pipeline:\nTest Loss: {test_loss:.4f}\nTest Acc: {test_accuracy:.4f}\nTest F1-score macro: {test_f1_score_macro:.4f}\nTest F1-score macro non-Neutral: {test_f1_m_ex:.4f}")

    logger.info("Run completed successfully.")

if __name__ == "__main__":
    main()