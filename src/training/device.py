import torch

def setup_device(config):
    use_cuda = config["use_cuda"]

    if use_cuda:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            raise RuntimeError(
                "Config is set to use CUDA, yet no CUDA available"
            )
    else:
        device = torch.device("cpu")

    use_amp = (
        use_cuda and
        config["use_amp"] == 1
    )

    scaler = (
        torch.amp.GradScaler("cuda")
        if use_amp else None
    )

    return device, use_amp, scaler