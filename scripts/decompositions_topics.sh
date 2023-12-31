#!/bin/bash

MODELS="bert gpt2 bloom"
DATASETS="wikitext_topics openwebtext_topics github_topics"
PROJ_DIR="out/topics"
SEED=1234

for MODEL in $MODELS
do
    for DATASET in $DATASETS
    do
        echo "Running decompositions on $DATASET $MODEL"
        IN="--init_from=$MODEL --dataset=$DATASET"
        OUT="--out_dir=${PROJ_DIR}/${DATASET}-${MODEL}"

        if [ "$MODEL" == "llama2" ]; then
            torchrun src/generate_decompositions_topics.py $IN $OUT \
                --batch_size=1 --n_batch=64
    
        else
            python src/generate_decompositions_topics.py $IN $OUT \
                --batch_size=64 --n_batch=1
        fi
    done
done