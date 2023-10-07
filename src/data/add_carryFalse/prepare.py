import os
import pickle
import numpy as np
import random
from tqdm import tqdm

chars = sorted(list("0123456789=+\n"))
vocab_size = len(chars)
print("all the unique characters:", "".join(chars))
print(f"vocab size: {vocab_size:,}")


# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]  # encoder: take a string, output a list of integers


def decode(l):
    "".join([itos[i] for i in l])  # decoder: take a list of integers, output a string


def generate_data(size, to_filename, digits_range):
    print("Generating data to", to_filename)

    def digits_to_str(digits):
        return "".join([str(d) for d in digits])

    dataset = []
    for i in range(size):
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

        if reverse:
            c = c[::-1]

        s = f"{a}+{b}={c}\n"
        ids = encode(s)
        dataset.append(ids)

        if i in np.linspace(0, size, num=10, dtype=int):
            print(s)

    with open(os.path.join(os.path.dirname(__file__), to_filename), "wb") as f:
        pickle.dump(dataset, f)

    return True


if __name__ == "__main__":
    # create the train and test splits
    train_n = 1000000
    val_n = 10000
    digits_range = [5, 10]
    reverse = True
    carry = True
    generate_data(
        size=train_n,
        to_filename=f"train_char.pkl",
        digits_range=digits_range,
    )
    generate_data(
        size=val_n,
        to_filename=f"val_char.pkl",
        digits_range=digits_range,
    )

    # save the meta information as well, to help us encode/decode later
    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
    }
    with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
