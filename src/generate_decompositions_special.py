"""
Perform the $h_{c,t} = \mu + pos_t + cvec_c + resid_{c,t}$ decomposition.
"""
import sys
import os
import torch
import numpy as np
from model import GPTConfig, GPT, Bert, RoFormer, Llama2, Bloom

# -----------------------------------------------------------------------------
init_from = "resume"
out_dir = "out"
dataset = "openwebtext"
dataset_suffix = ""
seed = 1234
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32' or 'bfloat16' or 'float16'
block_size = 512
batch_size = 64
n_batch = 100
n_cvec_batch = 1
id_start, id_end = 1, 512
ckpt = "ckpt"
embd_normalize = "none"
split = "train"
out_dir = "out"
special_vocab = [11, 13, 198, 262, 286, 284, 290, 257, 287, 447]

config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]

data_dir = os.path.join("src/data", dataset)
train_data = np.memmap(
    os.path.join(data_dir, f"train{dataset_suffix}"), dtype=np.uint16, mode="r"
)
val_data = np.memmap(
    os.path.join(data_dir, f"val{dataset_suffix}"), dtype=np.uint16, mode="r"
)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)


def get_batch(split, batch_size):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )

    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)

    return x, y


# ------------------------ Model -------------------------------
if init_from.startswith("gpt2"):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
elif init_from == "roformer":
    model = RoFormer()
elif init_from == "bert":
    model = Bert()
elif init_from == "llama2":
    model = Llama2(max_batch_size=batch_size, max_seq_len=block_size)
elif init_from == "bloom":
    model = Bloom()
else:
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, f"{ckpt}.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    dataset = checkpoint["config"]["dataset"]

    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)


# ------------------ Decomposition --------------------

from tqdm import tqdm

# data for pos basis and cbasis and special token
token_embd = {}
token_count = {}
for i in tqdm(range(n_batch), desc="Get data for pos and global mean"):
    x, y = get_batch(split, batch_size)
    with torch.no_grad():
        hiddens = model.generate_hiddens(x)  # (L, B, T, C)
        if i == 0:
            embeddings_mean_by_B = (hiddens / (n_batch * batch_size)).sum(1)
            embeddings_mean_by_T = (
                hiddens[:, :, id_start:id_end, :] / (id_end - id_start)
            ).sum(2)
        else:
            embeddings_mean_by_B += (hiddens / (n_batch * batch_size)).sum(1)
            embeddings_mean_by_T = np.array(
                [
                    np.vstack([new, old])
                    for old, new in zip(
                        embeddings_mean_by_T,
                        (hiddens[:, :, id_start:id_end, :] / (id_end - id_start)).sum(
                            2
                        ),
                    )
                ]
            )

        for i, token in enumerate(special_vocab):
            is_x = (x == token).numpy(force=True)
            if token not in token_embd:
                token_embd[token] = hiddens[:, is_x, :].sum(1)
                token_count[token] = is_x.sum()
            else:
                token_embd[token] += hiddens[:, is_x, :].sum(1)
                token_count[token] += is_x.sum()


L, T, C = embeddings_mean_by_B.shape
L, B, C = embeddings_mean_by_T.shape
Tp = id_end - id_start

global_mean_all, pos_basis_all, c_basis_all, token_all = [], [], [], []
mean_vec_by_B_all = np.memmap(
    os.path.join(
        out_dir, f"mean_vec_by_B_id{id_start}-{id_end}_{embd_normalize}_{split}.npy"
    ),
    dtype=np.float32,
    mode="w+",
    shape=(L, Tp, C),
)
embeddings = np.memmap(
    os.path.join(out_dir, "tmp_embd"),
    dtype=np.float32,
    mode="w+",
    shape=(n_cvec_batch * batch_size, L, T, C),
)
c_vecs_all = np.memmap(
    os.path.join(out_dir, f"cvec_id{id_start}-{id_end}_{embd_normalize}_{split}.npy"),
    dtype=np.float32,
    mode="w+",
    shape=(L, n_cvec_batch * batch_size * Tp, C),
)
resids_all = np.memmap(
    os.path.join(out_dir, f"resids_id{id_start}-{id_end}_{embd_normalize}_{split}.npy"),
    dtype=np.float32,
    mode="w+",
    shape=(L, n_cvec_batch * batch_size * Tp, C),
)

for layer_idx in tqdm(range(L), desc="Calculate pos and global mean"):
    mean_vec_by_B = embeddings_mean_by_B[layer_idx, id_start:id_end, :]
    mean_vec_by_T = embeddings_mean_by_T[layer_idx]
    global_mean = np.mean(mean_vec_by_B, axis=0, keepdims=True)
    pos_basis = mean_vec_by_B - global_mean
    c_basis = mean_vec_by_T - global_mean
    token = (
        np.array(
            [
                token_embd[token][layer_idx] / token_count[token]
                for token in special_vocab
            ]
        )
        - global_mean
    )

    global_mean_all.append(global_mean)
    pos_basis_all.append(pos_basis)
    c_basis_all.append(c_basis)
    token_all.append(token)

    mean_vec_by_B_all[layer_idx] = mean_vec_by_B


# data for c-vec
# We use a smaller batch size of n_cvec_batch because c-vecs contains
# values for all batch_size x L x T x C, which could be large.
# Therefore we choose a smaller n_cvec_batch to reduce the size.

for i in tqdm(range(n_cvec_batch), desc="Get data for cvec and resid"):
    x, y = get_batch(split, batch_size)
    with torch.no_grad():
        hiddens = model.generate_hiddens(x)

        for layer_idx in range(L):
            embeddings[
                i * batch_size : i * batch_size + batch_size, layer_idx
            ] = hiddens[layer_idx]


for layer_idx in tqdm(range(L), desc="Calculate cvec and resid"):
    c_vecs, resids = [], []
    for sample_idx in range(n_cvec_batch * batch_size):
        c_basis = (
            embeddings[sample_idx, layer_idx, id_start:id_end].mean(0)
            - global_mean_all[layer_idx]
        )

        c_vec = (
            embeddings[sample_idx, layer_idx, id_start:id_end, :]
            - pos_basis_all[layer_idx]
            - global_mean_all[layer_idx]
        )
        resid = c_vec - c_basis  # shaped (Tp, C)

        c_vecs_all[layer_idx, sample_idx * Tp : sample_idx * Tp + Tp] = c_vec
        resids_all[layer_idx, sample_idx * Tp : sample_idx * Tp + Tp] = resid


# -------------------------------- Save --------------------------------------
#
for obj, name in zip(
    [
        c_vecs_all,
        resids_all,
        mean_vec_by_B_all,
    ],
    [
        "cvec",
        "resids",
        "mean_vec_by_B",
    ],
):
    fname = f"{name}_id{id_start}-{id_end}_{embd_normalize}_{split}.npy"
    print("Saving a", obj.shape, "np.array to", os.path.join(out_dir, fname))


for obj, name in zip(
    [
        global_mean_all,
        pos_basis_all,
        c_basis_all,
        token_all,
    ],
    [
        "global_mean",
        "pos",
        "cbasis",
        "token",
    ],
):
    fname = f"{name}_id{id_start}-{id_end}_{embd_normalize}_{split}.npy"
    print("Saving a", np.array(obj).shape, "np.array to", os.path.join(out_dir, fname))
    np.save(
        os.path.join(out_dir, fname),
        np.array(obj),
    )
print("\n\n" + "#" * 50)
os.remove(os.path.join(out_dir, "tmp_embd"))
np.save(os.path.join(out_dir, "config"), config)
