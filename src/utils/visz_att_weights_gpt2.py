import seaborn as sns
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.model import GPT

torch.manual_seed(1234)
model = GPT.from_pretrained("gpt2")
L = model.config.n_layer
H = model.config.n_head
C = model.config.n_embd

sigma_multiplier = np.sqrt(2 * np.log(C**2))  # denoising parameter
top_num = 60  # how many top indices to show


def my_round(a, fac=1e3):
    return np.round(a * fac) / fac


def plot(layer_idx, pos, save_to=None):
    block = model.transformer.h[layer_idx]
    W_q, W_k, _ = block.attn.c_attn.weight.split(C, dim=0)
    W_q = W_q.T.view(C, H, C // H)
    W_k = W_k.T.view(C, H, C // H)

    W_QK = np.zeros((H, C, C))
    stats = np.zeros((L, H, 3))

    p = pos[layer_idx]
    u, s, vt = np.linalg.svd(p)

    nrow, ncol = 2, 6
    fig, axs = plt.subplots(nrow, ncol, figsize=(8 * ncol, 6 * nrow), dpi=100)
    for head in range(H):
        W_QK[head] = (W_q[:, head, :] @ W_k[:, head, :].T / np.sqrt(C // H)).numpy(
            force=True
        )
        W_QK_rot = vt @ (W_QK[head] - np.diag(np.diag(W_QK[head]))) @ vt.T
        stats[layer_idx, head, :] = np.array(
            [
                my_round(np.max(W_QK_rot.flatten())),
                my_round(np.min(W_QK_rot.flatten())),
                my_round(np.std(W_QK_rot.flatten())),
            ]
        )
        threshold = sigma_multiplier * stats[layer_idx, head, 2]
        indicator_sparse = np.abs(W_QK_rot) > threshold

        indicator_diagonal = np.eye(C, dtype=bool)
        W_QK_diagonal = W_QK[head] * indicator_diagonal
        W_QK_rot_sparse = W_QK_rot * indicator_sparse
        vmax1 = np.max(np.abs(W_QK_diagonal[:top_num, :top_num]))
        vmax2 = np.max(np.abs(W_QK_rot_sparse[:top_num, :top_num]))
        vmax = np.maximum(vmax1, vmax2)
        W_QK_diagonal[W_QK_diagonal == 0] = float("nan")
        W_QK_rot_sparse[W_QK_rot_sparse == 0] = float("nan")

        ax = axs[head // ncol, head % ncol]
        sns.heatmap(
            np.abs(W_QK_diagonal[:top_num, :top_num]),
            cmap="Reds",
            vmin=0,
            vmax=vmax,
            ax=ax,
        )
        sns.heatmap(
            np.abs(W_QK_rot_sparse[:top_num, :top_num]),
            cmap="Blues",
            vmin=0,
            vmax=vmax,
            ax=ax,
        )

        title = f"L{layer_idx}H{head}, "
        title += f"Max: {stats[layer_idx,head,0]}, min: {stats[layer_idx,head,1]}, std: {stats[layer_idx,head,2]}"
        ax.set_title(
            title,
            fontsize=10,
            weight="bold",
        )

    if save_to is not None:
        plt.savefig(save_to, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
