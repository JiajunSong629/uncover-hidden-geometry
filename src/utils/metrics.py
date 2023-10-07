from screenot.ScreeNOT import adaptiveHardThresholding
from tqdm import tqdm
import numpy as np
import scipy


def ranks_and_explained_ratios_and_relative_norm(pos, cvec, global_mean, up_bound=20):
    """
    pos is of shape (L, T, C), where L is the number of layers (plus 1),
    T is the length of positions, and C is the dimension of the positional embeddings.

    Returns
        np.array shaped (T, 4), columns screeNOT, stable rank, explained ratio, relative norm
    """
    L, T, C = pos.shape
    L, B, C = cvec.shape
    nseq = B // T

    result = np.zeros((L, 4))
    for l in tqdm(range(L)):
        p = pos[l]
        c = cvec[l]
        g = global_mean[l]
        p_full = np.concatenate([p for _ in range(nseq)])
        m = p_full + c + g

        psvals = scipy.linalg.svd(p)[1]
        rank = screenNOT(p, up_bound)

        result[l, 0] = rank
        result[l, 1] = stable_rank(psvals, k=0)
        result[l, 2] = explained_ratio(psvals, rank)
        result[l, 3] = scipy.linalg.norm(p_full, ord=2) / scipy.linalg.norm(m, ord=2)
    return result


def screenNOT(p, up_bound):
    return adaptiveHardThresholding(p, k=up_bound, strategy="i")[-1]


def stable_rank(svals, k=0):
    """
    Effective ranks

    Given k,
    rk = \sum_{j>k} \sigma_j^2 / \sigma_{k+1}^2
    """
    sec_mnt = svals**2
    cut = sec_mnt[k:]
    return (cut / sec_mnt[k]).sum()


def explained_ratio(svals, r):
    return svals[: int(r)].sum() / svals.sum()


def avg_similarity_between(context_basis, batch_size=64):
    L, B, C = context_basis.shape
    mask = np.kron(np.eye(B // batch_size), np.ones((batch_size, batch_size)))
    scores = []
    for layer_idx in range(L):
        c = context_basis[layer_idx]
        u, s, vt = scipy.linalg.svd(c)
        c = u[:, :20] @ np.diag(s[:20])
        c = c / np.sqrt(np.sum(c**2, axis=1, keepdims=True))
        score = np.where(mask == 0, np.abs(c @ c.T), 0)
        scores.append(np.sum(score) / np.sum(mask == 0))
    return np.array(scores)


def avg_similarity_within(context_basis, batch_size=64):
    L, B, C = context_basis.shape
    mask = np.kron(np.eye(B // batch_size), np.ones((batch_size, batch_size)))
    scores = []
    for layer_idx in range(L):
        c = context_basis[layer_idx]
        u, s, vt = scipy.linalg.svd(c)
        c = u[:, :20] @ np.diag(s[:20])
        c = c / np.sqrt(np.sum(c**2, axis=1, keepdims=True))
        score = np.where(mask == 1, np.abs(c @ c.T), 0)
        scores.append(np.sum(score) / np.sum(mask == 1))
    return np.array(scores)


def avg_similarity_between_NoPCA(context_basis, batch_size=64):
    L, B, C = context_basis.shape
    mask = np.kron(np.eye(B // batch_size), np.ones((batch_size, batch_size)))
    scores = []
    for layer_idx in range(L):
        c = context_basis[layer_idx]
        # u, s, vt = scipy.linalg.svd(c)
        # c = u[:, :20] @ np.diag(s[:20])
        c = c / np.sqrt(np.sum(c**2, axis=1, keepdims=True))
        score = np.where(mask == 0, np.abs(c @ c.T), 0)
        scores.append(np.sum(score) / np.sum(mask == 0))
    return np.array(scores)

def avg_similarity_within_NoPCA(context_basis, batch_size=64):
    L, B, C = context_basis.shape
    mask = np.kron(np.eye(B // batch_size), np.ones((batch_size, batch_size)))
    scores = []
    for layer_idx in range(L):
        c = context_basis[layer_idx]
        # u, s, vt = scipy.linalg.svd(c)
        # c = u[:, :20] @ np.diag(s[:20])
        c = c / np.sqrt(np.sum(c**2, axis=1, keepdims=True))
        score = np.where(mask == 1, np.abs(c @ c.T), 0)
        scores.append(np.sum(score) / np.sum(mask == 1))
    return np.array(scores)
