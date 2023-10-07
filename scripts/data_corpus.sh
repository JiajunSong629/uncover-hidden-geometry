DATASETS="shakespeare github openwebtext wikitext"
STREAMING_SAMPLES=1000
OVERWRITE="True"

for DATASET in $DATASETS
do
    python src/data/prepare_corpus.py \
        --dataset=$DATASET \
        --streaming_samples=$STREAMING_SAMPLES \
        --overwrite=$OVERWRITE
done