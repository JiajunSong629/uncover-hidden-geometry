DATASETS="shakespeare github openwebtext wikitext"
TOKENIZERS="gpt2+bert+bloom+llama2"
STREAMING_SAMPLES=1000
OVERWRITE="True"

for DATASET in $DATASETS
do
    python src/data/prepare_corpus.py \
        --dataset=$DATASET \
        --streaming_samples=$STREAMING_SAMPLES \
        --overwrite=$OVERWRITE \
        --tokenizers=$TOKENIZERS
done