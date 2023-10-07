#!bin/bash

SEED=1234
NOISE_LEVELS="0 1 2"

for NOISE_LEVEL in $NOISE_LEVELS
do
    python src/train_randomization.py conf/train_randomization.py \
        --seed=${SEED} \
        --noise_level=${NOISE_LEVEL} \
        --out_dir=out/randomization/noise_level_${NOISE_LEVEL} \
        --tensorboard_run_name=gpt${N_LAYER}L${N_HEAD}H-noiselevel${NOISE_LEVEL}-${SEED}
done