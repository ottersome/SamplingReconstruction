from typing import List

import numpy as np
import torch


def take_a_guess(tape, p0, p1):
    num = 0
    denum = 0
    for i in range(len(tape) - 1):
        from_state = tape[i]
        to_state = tape[i + 1]
        num += np.log(p0[from_state, to_state])
        denum += np.log(p1[from_state, to_state])
    return 0 if num > denum else 1


def take_guesses(
    paths: torch.Tensor,
    labels: torch.Tensor,
    p0: List[List[float]],
    p1: List[List[float]],
):
    nums = torch.zeros((paths.shape[0], 1))
    denums = torch.zeros((paths.shape[0], 1))


# A Bit better for memory
def multiplicity_guess(states, multiplicity, p0, p1):
    num = 0
    denum = 0
    if len(states) == 1:
        num += (multiplicity[0] - 1) * np.log(p0[states[0], states[0]])
        denum += (multiplicity[0] - 1) * np.log(p1[states[0], states[0]])
    else:
        for i, s in enumerate(states[1:]):
            # Take Multiplicity as same-state transition
            from_state = states[i]
            num += (multiplicity[i] - 1) * np.log(p0[from_state, from_state]) + np.log(
                p0[from_state, s]
            )
            denum += (multiplicity[i] - 1) * np.log(
                p1[from_state, from_state]
            ) + np.log(p1[from_state, s])

        num += (multiplicity[-1] - 1) * np.log(p0[states[-1], states[-1]])
        denum += (multiplicity[-1] - 1) * np.log(p1[states[-1], states[-1]])
    return 0 if num > denum else 1


def dummy(states, multiplicity, p0, p1):
    num = 0
    denum = 0
    for i, s in enumerate(states[1:]):
        # Take Multiplicity as same-state transition

        from_state = states[i]
        num += (multiplicity[i] - 1) * np.log(p0[from_state, from_state]) + np.log(
            p0[from_state, s]
        )
        denum += (multiplicity[i] - 1) * np.log(p1[from_state, from_state]) + np.log(
            p1[from_state, s]
        )
    return 0 if num > denum else 1
