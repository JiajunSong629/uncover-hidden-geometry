## Uncover hidden geometry in Transformers

## Getting Started

### Environment setup

```{bash}
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### (Optional) Llama2 checkpoints

To check the results of Llama2, you need to download the Llama2 checkpoints and place it under `src/llama2-ckpt`.


### Dataset

Run the following commands to prepare a subset of the corpus (choose from openwebtext, wikitext, github).
Edit the `DATASETS` entry in `scripts/data_corpus.sh` to specify. The generated datasets will be placed under `src/data/`.

```
chmod +x ./scripts/data_corpus.sh
./scripts/data_corpus.sh
```

## Inferencing on pretrained model


### Decomposition

Perform the $h_{c,t} = \mu + pos_t + cvec_c + resid_{c,t}$ decomposition on datasets and models. Edit the `DATASETS` entry in `scripts/data_corpus.sh` to specify the corpus or the models.

```{bash}
chmod +x ./scripts/decompositions.sh
./scripts/decompositions.sh
```


## Training

### NanoGPT

Run the following commands to train NanoGPT (6L6H384D) on character-level Shakespeare dataset.

```{bash}
python src/train.py conf/train_nanogpt.py
```

### Randomization Experiment

Run the following commands to train a Transformer (8L8H512D) with various levels of randomizations in the inputs.

```{bash}
chmod +x scripts/train_randomization.sh
./scripts/train_randomization.sh
```

### Arithmetic Experiment

Run the following commands to train a Transformer (8L8H512D) on an arithmetic task.

```{bash}
chmod +x scripts/train_randomization.sh
./scripts/train_addition.sh
```


## Reproducibility

The directory `notebooks/` contains jupyter notebooks that reproduce the figures and tables in the paper. Be aware that they may depend on datasets or checkpoints that requires generation in advance.