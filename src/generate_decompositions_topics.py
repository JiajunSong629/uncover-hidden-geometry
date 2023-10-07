# """
# This script generates positional context basis from a trained model.
# The result is used mainly for the visualization of context clustering plot,
# which is implemented in `analysis/figure_context_clustering.py`.
# """

import os
import torch
import numpy as np
from tqdm import tqdm
from src.model import GPT, Bert, Llama2, Bloom

dataset = "openwebtext_topics"
out_dir = "out/topics/openwebtext_topics/gpt2"
init_from = "gpt2"
id_start = 1
id_end = 128
block_size = 128
batch_size = 64
n_batch = 1
device = "cuda"

exec(open("configurator.py").read())  # overrides from command line or config file

# -----------------------------------------------------------------------------
if init_from.startswith("gpt2"):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
elif init_from == "bert":
    model = Bert()
elif init_from == "llama2":
    model = Llama2(max_batch_size=batch_size, max_seq_len=block_size)
elif init_from == "bloom":
    model = Bloom()
# -----------------------------------------------------------------------------
os.makedirs(out_dir, exist_ok=True)


def get_batch(data, batch_size):
    ix = np.linspace(0, len(data) - block_size, num=batch_size, dtype=int)
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    x = x.to(device)
    return x


data_dir = os.path.join("src/data", dataset)
datum = []
for file in os.listdir(data_dir):
    if file.endswith(f"_{init_from}.bin"):
        data = np.memmap(
            os.path.join(data_dir, file),
            dtype=np.uint16,
            mode="r",
        )
        datum.append(data)

# ------------------------------------------------------------------
data_rstrip = dataset[: -len(list("_topics"))]
load_dir = f"out/decompositions/{data_rstrip}-{init_from}"
embd_normalize = "none"
suffix = f"id1-512_{embd_normalize}_train.npy"
pos = np.load(os.path.join(load_dir, f"pos_{suffix}"))
L, T, C = pos.shape
mean_vec_by_B = np.memmap(
    os.path.join(load_dir, f"mean_vec_by_B_{suffix}"),
    mode="r",
    dtype=np.float32,
    shape=(L, T, C),
)

# -----------------------------------------------------------------

n_topics = len(datum)
n_samples = n_topics * batch_size * n_batch

p_basis = np.zeros((L, block_size - 1, C))
global_mean = np.zeros((L, C))
for layer in range(L):
    global_mean[layer] = mean_vec_by_B[layer, : block_size - 1].mean(0)
    p_basis[layer] = mean_vec_by_B[layer, : block_size - 1] - global_mean[layer]


# ----------------------------------------------------------------------
suffix = f"id{id_start}-{id_end}_{init_from}.npy"
embeddings = np.memmap(
    os.path.join(out_dir, f"embeddings_{suffix}"),
    mode="w+",
    dtype=np.float32,
    shape=(L, n_samples, block_size, C),
)
c_basis = np.memmap(
    os.path.join(out_dir, f"cbasis_{suffix}"),
    mode="w+",
    dtype=np.float32,
    shape=(L, n_samples, C),
)
c_vecs = np.memmap(
    os.path.join(out_dir, f"cvec_{suffix}"),
    mode="w+",
    dtype=np.float32,
    shape=(L, n_samples, id_end - id_start, C),
)

for i, data in enumerate(datum):
    for j in tqdm(range(n_batch), desc=f"Batch Progress at topic {i}"):
        ids = get_batch(data, batch_size)
        hiddens = model.generate_hiddens(ids)
        for layer in range(L):
            embeddings[
                layer,
                i * batch_size * n_batch
                + j * batch_size : i * batch_size * n_batch
                + j * batch_size
                + batch_size,
            ] = hiddens[layer]


# c_basis = np.zeros((L, n_samples, C))
# c_vecs = np.zeros((L, n_samples, id_end - id_start, C))
# resids = np.zeros((L, n_samples, id_end - id_start, C))
for layer in tqdm(range(L), desc="Layer progress"):
    c_basis[layer] = (
        embeddings[layer, :, id_start:id_end, :].mean(1) - global_mean[layer]
    )
    c_vecs[layer] = (
        embeddings[layer, :, id_start:id_end, :] - p_basis[layer] - global_mean[layer]
    )

# ------------------------------------------------------------------------------

for obj, name in zip([p_basis], ["pos"]):
    fname = f"{name}_id{id_start}-{id_end}_{init_from}.npy"
    print("Saving a", np.array(obj).shape, "np.array to", os.path.join(out_dir, fname))
    np.save(
        os.path.join(out_dir, fname),
        np.array(obj),
    )

for obj, name in zip([c_basis, c_vecs, embeddings], ["cbasis", "cvec", "embeddings"]):
    fname = f"{name}_id{id_start}-{id_end}_{init_from}.npy"
    print("Saving a", obj.shape, "np.array to", os.path.join(out_dir, fname))
