import logging
import os

import torch
import torch.nn.functional as F


def setup_logger(name: str, level=logging.INFO):
    cwd = os.getcwd()
    full_path = os.path.join(cwd, "./logs")
    os.makedirs(full_path, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    sh = logging.StreamHandler()
    fh = logging.FileHandler(os.path.join(full_path, name + ".log"), mode="w")
    sh.setLevel(level)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)

    logger.addHandler(sh)
    logger.addHandler(fh)

    return logger


def dec_rep_to_batched_rep(
    dec_tape: torch.Tensor,
    dec_periods: torch.Tensor,
    samp_budget: int,
    num_classes: int,
    add_position: bool,
):
    """
    Takes a representation of B samples all under the same sampling budget.
    Then expands into a one hot encoded representation that can be fed as a batch to the algorithm
    """
    device = dec_tape.device

    batch_size = dec_tape.shape[0]
    # Create Lengths
    lengths = 1 + (samp_budget - 1) * dec_periods
    max_len = torch.max(lengths)
    # Create Masks with lengths:
    masks = torch.zeros((dec_tape.shape[0], samp_budget), dtype=torch.float32)
    for i, length in enumerate(lengths):
        masks[i, :length] = 1
    # Create One Hot representations
    # All zeros means unknown. (Though I'm not sure this is the best way to specify it)
    one_hot = F.one_hot(dec_tape, num_classes=num_classes).float()  # +1 for padding

    full_resolution = torch.zeros(batch_size, max_len.item(), num_classes).to(
        device
    )  # type:ignore
    # Place the onehot decimated samples into full)resolutio
    for b in range(batch_size):  # CHECK: Proper behavior (should be fine)
        aranged_idx = torch.arange(
            0, (dec_periods[b] * (samp_budget - 1) + 1).item(), dec_periods[b].item()
        )
        full_resolution[b, aranged_idx] = one_hot[b, :].float()

    return full_resolution
