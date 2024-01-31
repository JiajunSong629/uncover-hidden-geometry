# Uncover hidden geometry in Transformers

## Getting Started

### Environment setup

```{bash}
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### (Optional) Llama2 checkpoints

To check the results of Llama2, you need to either place the checkpoints under `src/llama2-ckpt` or specify the path to the checkpoints.


### Prepare Dataset

Run the following commands to prepare a subset of the corpus (available options are `openwebtext`, `wikitext`, and `github`).
Edit the `DATASETS` entry in `scripts/data_corpus.sh` to specify. The generated datasets will be placed under `src/data/$DATASET`.

```
bash scripts/data_corpus.sh
```

## Uncover Hidden Geometry via Decomposition


Perform the $h_{c,t} = \mu + pos_t + cvec_c + resid_{c,t}$ decomposition on various datasets and models. Edit the `DATASETS` and `MODELS` entry in `scripts/decompositions.sh` to specify the corpus or the models.

```{bash}
bash scripts/decompositions.sh
```


### Reproducibility

The directory `notebooks/` contains jupyter notebooks that reproduce the figures and tables in the paper. Be aware that they may depend on datasets or checkpoints that need to be generated in advance.