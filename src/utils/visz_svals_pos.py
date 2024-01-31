from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def plot(pos, cvec, resid, save_to=None, only_pos=True):
    L, T, C = pos.shape
    L, S, C = cvec.shape
    nseq = S // T

    nrows, ncols = 2, 6

    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), dpi=100)
    for layer_idx in tqdm(range(L - 1), desc="Layer progress"):
        rdx, cdx = layer_idx // ncols, layer_idx % ncols
        ax = axs[rdx, cdx]
        p, c, r = pos[layer_idx], cvec[layer_idx], resid[layer_idx]
        p_full = np.concatenate([p for _ in range(nseq)])

        firstk = 60
        p_full, c, r = p_full[:T], c[:T], r[:T]
        p_svals = np.linalg.svd(p_full)[1][:firstk]
        ax.plot(np.arange(len(p_svals)), p_svals, marker=".", label="P")

        if not only_pos:
            c_svals = np.linalg.svd(c)[1][:firstk]
            r_svals = np.linalg.svd(r)[1][:firstk]

            ax.plot(np.arange(len(c_svals)), c_svals, marker=".", label="Cvec")
            ax.plot(np.arange(len(r_svals)), r_svals, marker=".", label="R")

        ax.set_title(f"Layer: {layer_idx}", weight="bold", fontsize=20)
        ax.set_yscale("log")
        ax.legend()

    fig.tight_layout()
    if save_to is not None:
        plt.savefig(save_to)
    else:
        plt.show()
