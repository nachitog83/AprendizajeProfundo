#!/usr/bin/env bash

set -ex

if [ ! -d "./data/meli-challenge-2019/" ]
then
    mkdir -p ./data
    echo >&2 "Downloading Meli Challenge Dataset"
    curl -L https://cs.famaf.unc.edu.ar/\~ccardellino/resources/diplodatos/meli-challenge-2019.tar.bz2 -o ./data/meli-challenge-2019.tar.bz2
    tar jxvf ./data/meli-challenge-2019.tar.bz2 -C ./data/
fi

if [ ! -f "./data/SBW-vectors-300-min5.txt.gz" ]
then
    mkdir -p ./data
    echo >&2 "Downloading SBWCE"
    curl -L https://cs.famaf.unc.edu.ar/\~ccardellino/resources/diplodatos/SBW-vectors-300-min5.txt.gz -o ./data/SBW-vectors-300-min5.txt.gz
fi

# Be sure the correct nvcc is in the path with the correct pytorch installation
export CUDA_HOME=/opt/cuda/10.1
export PATH=$CUDA_HOME/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
MOD=3
if [ $MOD == 1 ]
then
    python -m experiment.mlp \
        --train-data ./data/meli-challenge-2019/spanish.train.jsonl.gz \
        --token-to-index ./data/meli-challenge-2019/spanish_token_to_index.json.gz \
        --pretrained-embeddings ./data/SBW-vectors-300-min5.txt.gz \
        --language spanish \
        --validation-data ./data/meli-challenge-2019/spanish.validation.jsonl.gz \
        --embeddings-size 300 \
        --hidden-layers 512 256 128 \
        --dropout 0.3 \
        --epochs 3
    #    --batch-size 256 \
    #    --learning-rate 0.001 \
    #    --weight-decay 0.005
elif [ $MOD == 2 ]
then
    python -m experiment.rnn \
        --train-data ./data/meli-challenge-2019/spanish.train.jsonl.gz \
        --token-to-index ./data/meli-challenge-2019/spanish_token_to_index.json.gz \
        --pretrained-embeddings ./data/SBW-vectors-300-min5.txt.gz \
        --language spanish \
        --validation-data ./data/meli-challenge-2019/spanish.validation.jsonl.gz \
        --test-data ./data/meli-challenge-2019/spanish.test.jsonl.gz \
        --embeddings-size 300 \
        --hidden-layer 256 \
        --num-layers 2 \
        --dropout 0.2 \
        --bidirectional True \
        --epochs 5
    #    --batch-size 256 \
    #    --learning-rate 0.001 \
    #    --weight-decay 0.005
elif [ $MOD == 3 ]
then
    python -m experiment.cnn \
    --train-data './data/meli-challenge-2019/spanish.train.jsonl.gz' \
    --token-to-index './data/meli-challenge-2019/spanish_token_to_index.json.gz' \
    --pretrained-embeddings './data/SBW-vectors-300-min5.txt.gz' \
    --language 'spanish' \
    --validation-data './data/meli-challenge-2019/spanish.validation.jsonl.gz' \
    --test-data './data/meli-challenge-2019/spanish.test.jsonl.gz' \
    --embeddings-size 300 \
    --epochs 8 \
    --batch-size 128 \
    --random-buffer-size 2048 \
    --freeze-embedings True \
    --lr 1e-3 \
    --weight-decay 1e-6 \
    --filters-count 100 \
    --filters-width 2 3 4 \
    --dimensions 128
fi