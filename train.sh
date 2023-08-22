#!/bin/bash

MODEL_NAME = "nghuyong/ernie-2.0-base-en"
ALPHA = 0.1
DATASET = "msmarco"
FEATURES = "coverage"

python train_cross-encoder_msmarco.py \
--model_name $MODEL_NAME \
--alpha $ALPHA \
--dataset $DATASET \
--input_features $FEATURES
