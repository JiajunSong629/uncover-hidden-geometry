import random
import numpy as np
from src.utils import load_model
import torch

chars = sorted(list("0123456789=+\n"))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]  # encoder: take a string, output a list of integers


def decode(l):
    "".join([itos[i] for i in l])  # decoder: take a list of integers, output a string


def digits_to_str(digits):
    return "".join([str(d) for d in digits])


def make_nocarry(digits_range):
    l, h = digits_range
    # Restrict to equal lengths addition
    n1 = random.randint(l, h)
    n2 = random.randint(n1, h)

    a_digits = [random.randint(0, 9) for _ in range(n1)]
    b_digits = [random.randint(0, 9) for _ in range(n2 - n1)] + [
        random.randint(0, 9 - adigit) for adigit in a_digits
    ]
    c_digits = b_digits[: (n2 - n1)] + [
        ad + bd for ad, bd in zip(a_digits, b_digits[n2 - n1 : n2])
    ]

    if random.uniform(0, 1) > 0.5:
        a_digits, b_digits = b_digits, a_digits

    a = digits_to_str(a_digits)
    b = digits_to_str(b_digits)
    c = digits_to_str(c_digits)
    c = c[::-1]

    return a, b, c


def make_carry(digits_range):
    l, h = digits_range
    succeed = False
    len_a = random.randint(l, h)
    len_b = random.randint(l, h)
    while not succeed:
        a = random.randint(np.power(10, len_a - 1), np.power(10, len_a) - 1)
        b = random.randint(np.power(10, len_b - 1), np.power(10, len_b) - 1)
        c = a + b
        len_c = len(str(c))
        if l <= len_c and len_c <= h:
            succeed = True

    a, b, c = str(a), str(b), str(c)
    c = c[::-1]
    return a, b, c


if __name__ == "__main__":
    carry_model = load_model.load("out/add_carryTrue/ckpt.pt")
    nocarry_model = load_model.load("out/add_carryFalse/ckpt.pt")

    in_range, out_range = [5, 10], [1, 4]

    def acc(model, digits_range, sample_func, N=1000):
        corct = 0
        for _ in range(N):
            a, b, c = sample_func(in_range)
            s = f"{a}+{b}="
            idx = torch.tensor(encode(s)).unsqueeze(0)
            sol = carry_model.generate(idx, max_new_tokens=len(c), top_k=1)
            sol_c = sol[len(s) :]

            if sol_c == c:
                corct += 1
            else:
                print(a, b, sol_c, c)
        return corct / N
