import torch
import logging
from src.dataloader.dataloader import load_tokenizer, build_conversation_dataloader
from src.models.conversation_classifier import ConversationClassifier
from src.models.bert_embedding import BERTEmbedding
from src.training.loops import train_model
from src.training.checkpoint import load_model
from src.training.device import setup_device
from src.training.metrics import get_final_test_accuracy
from src.training.debug import debug_nan

logger = logging.getLogger(__name__.split(".")[-1])

def run_conversation_pipeline(config, paths):
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

    train_loader = build_conversation_dataloader(
        train_data,
        batch_size=config["conversation_recognition"]["batch_size"],
        do_shuffling=True
    )

    val_loader = build_conversation_dataloader(
        val_data,
        batch_size=config["conversation_recognition"]["batch_size"],
        do_shuffling=False
    )

    test_loader = build_conversation_dataloader(
        test_data,
        batch_size=config["conversation_recognition"]["batch_size"],
        do_shuffling=False
    )

    logger.info(f"Data loaded: {len(train_loader.dataset)} train samples, {len(val_loader.dataset)} val samples, {len(test_loader.dataset)} test samples.")

    # Get embedding
    embedding = BERTEmbedding(bert_config=config["bert"])
    # Assume that utterance pipeline has been ran, if not or config specified: init
    checkpoint_path = paths.checkpoints / config["utterance_recognition"]["checkpoint_name"] # Path to the utterance classifier checkpoint, which is used to initialize the embedding layer of the conversation classifier
    if (not checkpoint_path.exists() or config["conversation_recognition"]["embed_use"] == 0):
        logger.info("Utterance checkpoint not found or embed_use is set to 0, initializing embedding with pretrained BERT weights.")
        pass
    else:
        logger.info(f"Loading utterance checkpoint from {checkpoint_path} to initialize embedding.")
        missing, unexpected = embedding.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
        # 1. Check for things you EXPECTED to be missing (your new classifier head)
        print("\nMissing keys (should only be your new head):", missing)

        # 2. Check for things you EXPECTED to be ignored (the old classifier head)
        print("\nUnexpected keys (should only be the old head):", unexpected)

    # Build model
    model = ConversationClassifier(
        embedding=embedding,
        dataset_config=config["dataset"][config["dataset_name"]], 
        attention_config=config["attention"],
        head_config=config["conversation_head"],
    ).to(device)

    model_path = (
        paths.saved_models / config["final_model_name"]
    )

    # Train
    if (not model_path.exists() or config["conversation_recognition"]["retrain"]):
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader, 
            config=config,
            running_pipeline="conversation_recognition",
            model_path=model_path,
            device=device,
            use_amp=use_amp,
            scaler=scaler
        )
    
    load_model(model, model_path, config["compile_model"], device)

    print("Checking if the model has any NaN parameter\n")
    debug_nan(model)

    # Final test with test_data
    test_loss, test_accuracy, test_f1_m, test_f1_m_ex = get_final_test_accuracy(model, test_loader, device)

    # Return test_loss and test_accurcacy
    return test_loss, test_accuracy, test_f1_m, test_f1_m_ex