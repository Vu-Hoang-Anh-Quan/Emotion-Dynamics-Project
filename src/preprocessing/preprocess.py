from datasets import load_dataset

def load_current_dataset(dataset_name):
    # load the corresponding dataset specified in config
    dataset = load_dataset(dataset_name, revision="refs/convert/parquet")
    return dataset

def split_data(current_dataset):
    # change the training dataset to the desirable result before return it to further process
    processed = []

    for conversation in current_dataset:
        utterances = conversation["dialog"]
        emotions = conversation["emotion"]

        for i, (utt, emo) in enumerate(zip(utterances, emotions)):
            sample = {
                "text": utt,
                "speaker": i % 2,   # 0 or 1
                "emotion": emo      # integer label
            }
            processed.append(sample)

    return processed
    
def preprocess_data(config):
    dataset = load_current_dataset(config["dataset_name"])
    train_data = split_data(dataset['train'])
    val_data = split_data(dataset['validation'])
    test_data = split_data(dataset['test'])
    return [train_data, val_data, test_data]
