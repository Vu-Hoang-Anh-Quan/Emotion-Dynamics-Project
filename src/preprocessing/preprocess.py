from datasets import load_dataset

def load_current_dataset(dataset_name):
    # load the corresponding dataset specified in config
    dataset = load_dataset(dataset_name, revision="refs/convert/parquet")
    return dataset

def process_data(current_dataset, prev_k):
    # change the training dataset to the desirable result before return it to further process
    processed = []

    for conversation in current_dataset:
        utterances = conversation["dialog"]
        emotions = conversation["emotion"]

        for i in range(len(utterances)):
            context_turns = []

            for j in range(i-1, max(0, i-prev_k)-1, -1):
                prev_utt = utterances[j]
                prev_speaker = j % 2

                context_turns.append((prev_utt, prev_speaker))

            utt = utterances[i]
            emo = emotions[i]
            sample = {
                "text": utt,
                "speaker": i % 2,   # 0 or 1
                "emotion": emo      # integer label
            }
            processed.append({
                "context": context_turns,
                "current": sample
            })

    return processed
    
def preprocess_data(config):
    k_prev = config["consider_previous_k"]

    dataset = load_current_dataset(config["dataset_name"])
    train_data = process_data(dataset['train'], k_prev)
    val_data = process_data(dataset['validation'], k_prev)
    test_data = process_data(dataset['test'], k_prev)
    return [train_data, val_data, test_data]
