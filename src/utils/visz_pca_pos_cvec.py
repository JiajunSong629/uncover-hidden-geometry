import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


n_plot_seq = 10
nrows, ncols = 2, 6


def plot(pos, cvec, plot_context=True, save_to=None):
    L, T, C = pos.shape

    if L == 13:
        selected_layers = range(L - 1)
    else:
        selected_layers = list(np.linspace(0, L - 2, num=12, dtype=int))

    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), dpi=100)

    for subplot_idx, layer_idx in enumerate(
        tqdm(selected_layers, desc="Layer progress")
    ):
        rdx, cdx = subplot_idx // ncols, subplot_idx % ncols
        p = pos[layer_idx]

        # apply PCA
        u, s, vt = np.linalg.svd(p)
        proj_mat = vt[:2, :].T
        pc = p @ proj_mat

        colors_blue = [plt.cm.Blues(x) for x in np.linspace(0.3, 1, T)]

        ax = axs[rdx, cdx]
        ax.scatter(pc[:, 0], pc[:, 1], c=colors_blue, label="pos")

        if plot_context:
            c = cvec[layer_idx, : n_plot_seq * T]
            pc2 = c @ proj_mat
            colors_red = np.array(
                [plt.cm.Reds(x) for x in np.linspace(0.3, 1, T)] * n_plot_seq
            )
            ax.scatter(pc2[:, 0], pc2[:, 1], c=colors_red, alpha=0.1, label="c-vec")

        ax.set_title(f"Layer: {layer_idx}", weight="bold", fontsize=30)

    for ax in axs.ravel():
        ax.set_axis_off()
    fig.tight_layout()

    if save_to is not None:
        plt.savefig(save_to, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
