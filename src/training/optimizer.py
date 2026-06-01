import torch

NO_DECAY = [
    "bias",
    "LayerNorm.weight"
]


def get_optimizer(model, config):

    optimizer_groups = []

    groups = model.optimizer_groups()

    for group_name, module in groups.items():

        decay_params = []
        no_decay_params = []

        for name, param in module.named_parameters():

            if not param.requires_grad:
                continue

            if any(
                nd in name
                for nd in NO_DECAY
            ):
                no_decay_params.append(param)

            else:
                decay_params.append(param)

        lr = config[group_name]["lr"]

        if decay_params:
            optimizer_groups.append({
                "params": decay_params,
                "lr": lr,
                "weight_decay": config["weight_decay"]
            })

        if no_decay_params:
            optimizer_groups.append({
                "params": no_decay_params,
                "lr": lr,
                "weight_decay": 0.0
            })

    return torch.optim.AdamW(
        optimizer_groups
    )