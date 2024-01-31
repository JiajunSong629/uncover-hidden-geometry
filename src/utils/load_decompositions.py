import os
import numpy as np


def load_pos_cvec_global_mean(dataset, model, root_dir="out/decompositions"):
    data_dir = os.path.join(root_dir, f"{dataset}-{model}")
    suffix = "id1-512_none_train.npy"
    global_mean = np.load(os.path.join(data_dir, f"global_mean_{suffix}"))
    pos = np.load(os.path.join(data_dir, f"pos_{suffix}"))

    L, T, C = pos.shape
    N = 8 * T if model == "llama2" else 32 * T
    cvec = np.memmap(
        os.path.join(data_dir, f"cvec_{suffix}"),
        mode="r",
        shape=(L, N, C),
        dtype=np.float32,
    )
    return pos, cvec, global_mean


def load_pos_cvec_resid(dataset, model, root_dir="out/decompositions"):
    data_dir = os.path.join(root_dir, f"{dataset}-{model}")
    suffix = "id1-512_none_train.npy"
    pos = np.load(os.path.join(data_dir, f"pos_{suffix}"))

    L, T, C = pos.shape
    N = 8 * T if model == "llama2" else 32 * T
    cvec = np.memmap(
        os.path.join(data_dir, f"cvec_{suffix}"),
        mode="r",
        shape=(L, N, C),
        dtype=np.float32,
    )
    resid = np.memmap(
        os.path.join(data_dir, f"resids_{suffix}"),
        mode="r",
        shape=(L, N, C),
        dtype=np.float32,
    )
    return pos, cvec, resid


def load_pos(dataset, model, root_dir="out/decompositions"):
    data_dir = os.path.join(root_dir, f"{dataset}-{model}")
    suffix = "id1-512_none_train.npy"
    pos = np.load(os.path.join(data_dir, f"pos_{suffix}"))

    return pos


def load_pos_cbasis(dataset, model, root_dir):
    data_dir = os.path.join(root_dir, f"{dataset}-{model}")
    suffix = "id1-512_none_train.npy"
    pos = np.load(os.path.join(data_dir, f"pos_{suffix}"))
    cbasis = np.load(os.path.join(data_dir, f"cbasis_{suffix}"))

    return pos, cbasis
