#!/bin/bash

MODELS="gpt2 bert bloom llama2"
DATASETS="wikitext openwebtext github"
PROJ_DIR="out/decompositions"
SEED=1234

for MODEL in $MODELS
do
    for DATASET in $DATASETS
    do
        echo "Running decompositions on $DATASET $MODEL"
        IN="--init_from=$MODEL --dataset=$DATASET --dataset_suffix=_$MODEL.bin"
        OUT="--out_dir=${PROJ_DIR}/${DATASET}-${MODEL}"

        if [ "$MODEL" == "llama2" ]; then
            torchrun src/generate_decompositions.py conf/llama2_decompositions.py \
                --seed=$SEED $IN $OUT
    
        else
            python src/generate_decompositions.py conf/default_decompositions.py \
                --seed=$SEED $IN $OUT
        fi
    done
done