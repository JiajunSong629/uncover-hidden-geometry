DATASETS="github_topics openwebtext_topics wikitext_topics"
TOKENIZERS="gpt2+bert+bloom+llama2"
OVERWRITE="True"

for DATASET in $DATASETS
do
    python src/data/prepare_topics.py \
        --overwrite=$OVERWRITE \
        --dataset=$DATASET \
        --tokenizers=$TOKENIZERS
done