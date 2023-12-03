import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags
from tqdm import tqdm

from ..statistics.statistics import trans_matrix


# Extremely ad-hoc
def show_sanity_matrxs(matrices, titles):
    # Show Transition Matrices
    fig, ax = plt.subplots(1, len(matrices))
    fig.subplots_adjust(top=1.30)
    fig.suptitle(titles[-1])
    labels = ["True", "Empirical"]
    for m, data in enumerate(matrices):
        ax[m].title.set_text(titles[m])
        if hasattr(data, "shape") and len(data.shape) == 2:
            ax[m].imshow(data)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    ax[m].text(
                        j,
                        i,
                        "%.2f " % data[i, j],
                        ha="center",
                        va="center",
                        color="w",
                        fontsize=6,
                    )
        else:  # Assume its 1d data
            for i, bar in enumerate(data):
                ax[m].bar(
                    np.arange(0, len(bar), dtype=float) + (0.40 * i),
                    bar,
                    0.40,
                    label=labels[i],
                )
            ax[m].legend()
            state_dist = ax[m].set_title(titles[m])

    plt.show()
    plt.close()


def get_q_mat(rates, shape):
    shape += 1
    k = [
        rates["mu"] * np.ones(shape - 1),
        -(rates["mu"] + rates["lam"]) * np.ones(shape),
        rates["lam"] * np.ones(shape - 1),
    ]
    offset = [-1, 0, 1]
    tridiag = diags(k, offset).toarray()
    tridiag[0, 0] = -rates["lam"]
    tridiag[-1, -1] = -rates["mu"]
    return tridiag


def show_trans_matrx(holdTimes_tape, state_tape):
    # Show Transition Matrices
    fig, ax = plt.subplots(1, 1)
    trans = trans_matrix(holdTimes_tape, state_tape)
    plt.imshow(trans)
    for i in range(trans.shape[0]):
        for j in range(trans.shape[1]):
            ax.text(
                j,
                i,
                "%.2f " % trans[i, j],
                ha="center",
                va="center",
                color="w",
                fontsize=4,
            )
    ax.set_title("Preliminary Transition transrix")
    plt.show()
    plt.close()


def save_array_of_pictures(axs, thresholds, x_axis, varying_y_axis, path, name):
    # New
    if len(x_axis) != len(varying_y_axis):
        raise IOError
    # Create Dir if not existant
    os.makedirs(os.path.abspath(path), exist_ok=True)

    # Just a bit of caution
    fin_image_name = path
    if path[-1] == "/":
        fin_image_name += "/" + name
    fin_image_name += name

    # Num Zeros
    z = 1
    while len(varying_y_axis) / (10**z) > 0:
        z += 1
    point_sizes = np.exp(4 * np.array(thresholds) / np.max(thresholds))

    for i in tqdm(range(len(varying_y_axis))):
        axs.scatter(x_axis[:i], varying_y_axis[:i], point_sizes[:i], color="b")
        plt.savefig((fin_image_name + f"{i:05d}").format(i), dpi=100)


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--length",
        dest="length",
        default=10000,
        type=int,
        help="Length of episode in discrete realizations.",
    )
    parser.add_argument(
        "--mu", dest="mu", default=0.15, type=float, help="Service Rate"
    )
    parser.add_argument(
        "--lambda", dest="lam", default=0.10, type=float, help="Birth Rate"
    )
    parser.add_argument(
        "--samprate",
        dest="samprate",
        default=1.0,
        type=float,
        help="Rate at which we sample real line.",
    )
    parser.add_argument(
        "--state_limit",
        dest="state_limit",
        default=-1,
        type=int,
        help="Make the State Space Limited in Positive Integer Space by Providing Max Value.",
    )
    parser.add_argument(
        "--detection_guesses",
        dest="detection_guesses",
        default=-100,
        type=int,
        help="How many guesses we will take for detection",
    )
    parser.add_argument(
        "--init_state",
        dest="init_state",
        type=int,
        default=0,
        help="Initial State in the real line.(Amnt of current events)",
    )
    parser.add_argument(
        "--show_sanity_check",
        dest="show_sanity_check",
        type=bool,
        default=False,
        help="Shows the transition matrix estimated from  continuous labels.",
    )
    parser.add_argument(
        "--method",
        dest="method",
        choices=["event_driven_mle", "log_mat", "fixed_delta_t"],
        default="fixed_sampled_rate",
        help="Initial State in the real line.(Amnt of current events)",
    )
    parser.add_argument(
        "--preload_gmat_loc",
        dest="preload_gmat_loc",
        type=str,
        default="./secondary_tools/complex_generator.npy",
        help="Initial State in the real line.(Amnt of current events)",
    )
    parser.add_argument(
        "--num_samples",
        dest="num_samples",
        type=int,
        default=None,
        help="Number of Samples when sampling",
    )
    parser.add_argument(
        "--xres", dest="xres", type=int, default=1000, help="Resolution of X axis"
    )
    parser.add_argument(
        "--figtitle",
        dest="figtitle",
        type=str,
        default=None,
        help="Title for resulting figure",
    )

    return parser.parse_args()
