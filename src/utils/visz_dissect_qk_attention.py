import torch.nn.functional as F
import torch
import math
import matplotlib.pyplot as plt
import seaborn as sns


def custom_hm(data, vmin, vmax, ax, title):
    sns.heatmap(
        data,
        vmin=vmin,
        vmax=vmax,
        cmap="coolwarm",
        ax=ax,
        square=True,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"shrink": 0.6},
    )
    ax.tick_params(left=False, bottom=False)
    ax.set_title(title, weight="bold", fontsize=15)


@torch.no_grad()
def dissect_attentions(ids, model, pos, global_mean):
    b, t = ids.size()
    model.eval()
    outputs = torch.tensor(model.generate_hiddens(ids))

    QKs = []
    for layer_idx, (block, h) in enumerate(zip(model.transformer.h, outputs[:-1])):
        h = h[:, 1:]
        h_pos = torch.tensor(pos[layer_idx][: t - 1]).unsqueeze(0)
        h_cvec = h - h_pos  # - global_mean[layer_idx][: t - 1]

        B, T, C = h_pos.shape
        H = 12

        QK = {}
        names = ["pos", "cvec"]
        for i, h_q in enumerate([h_pos, h_cvec]):
            for j, h_k in enumerate([h_pos, h_cvec]):
                h_q = block.ln_1(h_q)
                h_k = block.ln_1(h_k)
                q, _, _ = block.attn.c_attn(h_q).split(C, dim=2)
                _, k, _ = block.attn.c_attn(h_k).split(C, dim=2)
                k = k.view(B, T, H, C // H).transpose(1, 2)  # (B, nh, T, hs)
                q = q.view(B, T, H, C // H).transpose(1, 2)  # (B, nh, T, hs)

                qk = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

                QK[names[i] + "-" + names[j]] = qk

        h = block.ln_1(h)
        q, _, _ = block.attn.c_attn(h).split(C, dim=2)
        _, k, _ = block.attn.c_attn(h).split(C, dim=2)
        k = k.view(B, T, H, C // H).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, H, C // H).transpose(1, 2)  # (B, nh, T, hs)
        qk = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        mask = torch.tril(torch.ones(T, T).view(1, 1, T, T))
        att = qk.masked_fill(mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        QK["total"] = qk
        QK["att"] = att

        QKs.append(QK)

    return QKs


# (layer, head) = (10, 7), (5,1) induction-head
# (layer, head) = (0, 1)  self-token
# (layer, head) = (4, 11) previous token

# model_name = "gpt2"
# save_dir = os.path.join(os.path.dirname(__file__), "save_objects")
# out_dir = "measurements/section4/figures"
# os.makedirs(out_dir, exist_ok=True)
# # with open(f"{save_dir}/QKs_{model_name}_rm.pkl", "rb") as f:
# #     QKs = pickle.load(f)

# QKs = attentions(ids, remove_global_mean=True)
# QKs_rm = attentions(ids, remove_global_mean=True)


def plot(QKs, layer, head, save_to=None):
    names = ["pos", "cvec"]
    nrows, ncols = 1, 6

    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), dpi=100)
    vmax = float("-inf")
    vmin = float("inf")
    for i in range(2):
        for j in range(2):
            ax = axs[2 * i + j]
            name = names[i] + "-" + names[j]
            vmax = max(vmax, QKs[layer][name][0, head].max())
            vmin = min(vmin, QKs[layer][name][0, head].min())

    for i in range(2):
        for j in range(2):
            ax = axs[2 * i + j]
            name = names[i] + "-" + names[j]
            custom_hm(
                QKs[layer][name][0, head], vmin, vmax, ax, f"QK component: {name}"
            )
    custom_hm(QKs[layer]["total"][0, head], vmin, vmax, axs[4], f"total QK")
    custom_hm(QKs[layer]["att"][0, head], 0, 1, axs[5], "attention")

    if save_to is not None:
        plt.savefig(save_to, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
