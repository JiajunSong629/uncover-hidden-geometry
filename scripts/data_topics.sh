DATASETS="github_topics openwebtext_topics wikitext_topics"
OVERWRITE="True"

for DATASET in $DATASETS
do
    python src/data/prepare_topics.py \
        --overwrite=$OVERWRITE \
        --dataset=$DATASET
done