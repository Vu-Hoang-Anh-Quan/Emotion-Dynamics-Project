from datasets import load_dataset

def load_current_dataset(dataset_name):
    # load the corresponding dataset specified in config
    dataset = load_dataset(dataset_name, revision="refs/convert/parquet")
    return dataset

def process_data(current_dataset):
    # change the training dataset to the desirable result before return it to further process
    processed = []

    for conversation in current_dataset:
        utterances = conversation["dialog"]
        emotions = conversation["emotion"]
        
        speaker_aware_utterances = []
        for i in range(len(utterances)):
            speaker_aware_utterances.append(
                f"S{i%2}: {utterances[i]}"
            )
        processed.append({
            "utterances": speaker_aware_utterances,
            "labels": emotions
        })
    return processed
    
def preprocess_data(config):
    dataset = load_current_dataset(config["dataset_name"])
    train_data = process_data(dataset['train'])
    val_data = process_data(dataset['validation'])
    test_data = process_data(dataset['test'])
    return [train_data, val_data, test_data]
