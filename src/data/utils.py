import os
import requests
import pickle
from datasets import load_dataset
from transformers import (
    GPT2TokenizerFast,
    LlamaTokenizer,
    BertTokenizerFast,
    BloomTokenizerFast,
)


def download_dataset(dataset, streaming_samples, input_file_path):
    if dataset == "github":
        ds = load_dataset(
            "codeparrot/github-code",
            streaming=True,
            split="train",
            languages=["Python"],
        )
        key = "code"
    elif dataset == "openwebtext":
        ds = load_dataset(
            "Skylion007/openwebtext",
            streaming=True,
            split="train",
        )
        key = "text"
    elif dataset == "wikitext":
        ds = load_dataset(
            "wikitext",
            "wikitext-103-v1",
            streaming=True,
            split="train",
        )
        key = "text"
    elif dataset == "shakespeare":
        print("Downloading", dataset)
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_file_path, "w") as f:
            f.write(requests.get(data_url).text)
    else:
        raise ValueError(
            f"{dataset} not supported! Must be in 'github', 'openwebtext', 'wikitext'."
        )

    if dataset in ["github", "wikitext", "openwebtext"]:
        print("Downloading", dataset)
        samples = []
        for sample in ds.take(streaming_samples):
            samples.append(sample[key])
        with open(input_file_path, "w") as f:
            f.write("\n".join(samples))


def char_handler(data):
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    ids = [stoi[c] for c in data]

    # save the meta information as well, to help us encode/decode later
    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
    }
    return ids, meta


def tokenize(tokenizers, data, data_dir):
    tokenized_data = {}
    for tokenizer, model_name in zip(
        [
            GPT2TokenizerFast.from_pretrained("gpt2"),
            BertTokenizerFast.from_pretrained("bert-base-cased"),
            LlamaTokenizer.from_pretrained(
                os.path.join(os.path.dirname(__file__), "llama2.tokenizer")
            ),
            BloomTokenizerFast.from_pretrained("bigscience/bloom-560m"),
            None,
        ],
        [
            "gpt2",
            "bert",
            "llama2",
            "bloom",
            "char",
        ],
    ):
        if model_name not in tokenizers:
            continue

        print(f"Tokenize with {model_name} tokenizer...")
        if model_name == "char":
            ids, meta = char_handler(data)
            with open(os.path.join(data_dir, "meta.pkl"), "wb") as f:
                pickle.dump(meta, f)
        else:
            ids = tokenizer(data, return_tensors="pt")["input_ids"][0]

        tokenized_data[model_name] = ids

    return tokenized_data
