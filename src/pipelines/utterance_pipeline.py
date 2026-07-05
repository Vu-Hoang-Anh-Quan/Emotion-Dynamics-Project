import torch
import logging
from src.dataloader.dataloader import load_tokenizer, build_utterance_dataloader
from src.models.utterance_classifier import UtteranceClassifier
from src.training.loops import train_model
from src.training.checkpoint import load_model, save_model
from src.training.device import setup_device
from ..utils.plotter import TrainingPlotter

logger = logging.getLogger(__name__.split(".")[-1])
from src.training.metrics import get_final_test_accuracy

def run_utterance_pipeline(config, paths):
    device, use_amp, scaler = setup_device(config)

    load_tokenizer()

    data_dir = paths.data

    train_data = torch.load(
        data_dir / "train_tokenized.pt",
        map_location=device
    )

    val_data = torch.load(
        data_dir / "val_tokenized.pt",
        map_location=device
    )

    test_data = torch.load(
        data_dir / "test_tokenized.pt",
        map_location=device
    )

    train_loader = build_utterance_dataloader(
        train_data,
        batch_size=config["utterance_recognition"]["batch_size"],
        do_shuffling=True
    )

    val_loader = build_utterance_dataloader(
        val_data,
        batch_size=config["utterance_recognition"]["batch_size"],
        do_shuffling=False
    )

    test_loader = build_utterance_dataloader(
        test_data,
        batch_size=config["utterance_recognition"]["batch_size"],
        do_shuffling=False
    )

    logger.info(f"Data loaded: {len(train_loader.dataset)} train samples, {len(val_loader.dataset)} val samples, {len(test_loader.dataset)} test samples.")

    # Build model
    model = UtteranceClassifier(
        dataset_config=config["dataset"][config["dataset_name"]],
        bert_config=config["bert"],
    ).to(device)

    model_path = (
        paths.checkpoints
        / config["utterance_recognition"]["model_name"]
    )

    plotter = TrainingPlotter(save_dir=paths.checkpoints / "training_plots", filename="utterance_training_curves.png")

    # Train
    if (not model_path.exists() or config["utterance_recognition"]["retrain"]):
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            running_pipeline="utterance_recognition",
            model_path=model_path,
            plotter=plotter,
            device=device,
            use_amp=use_amp,
            scaler=scaler
        )

    # Load the best model
    load_model(model, model_path, config["compile_model"], device)

    # Final test with test_data
    test_loss, test_accuracy, test_f1_m, test_f1_m_ex = get_final_test_accuracy(model, test_loader, device, config["utterance_recognition"])

    # Save just the embedding model to checkpoint path
    checkpoint_path = paths.checkpoints / config["utterance_recognition"]["checkpoint_name"]
    save_model(model.embedding, checkpoint_path)

    # Return test_loss and test_accurcacy
    return test_loss, test_accuracy, test_f1_m, test_f1_m_ex