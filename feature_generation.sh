#!/bin/bash
conda activate sentence_transformers
conda init bash
DATASET='msmarco'
NUM_GPUS=1

# python prepare_data.py --dataset $DATASET --pos_neg_ratio 4 --max_samples 1000

# #Convert to sentences
# python convert_to_sentences.py --dataset $DATASET

# conda activate triple
# cd openie6
# pip install --upgrade gevent

# #Triple extraction
# python run.py --mode splitpredict --inp data/$DATASET/queries_sentences.txt --out data/$DATASET/queries_triples.txt --rescoring --task oie --gpus $NUM_GPUS --oie_model models/oie_model/epoch=14_eval_acc=0.551_v0.ckpt --conj_model models/conj_model/epoch=28_eval_acc=0.854.ckpt --rescore_model models/rescore_model --num_extractions 5 
# python run.py --mode splitpredict --inp data/$DATASET/passages_sentences.txt --out data/$DATASET/passages_triples.txt --rescoring --task oie --gpus $NUM_GPUS --oie_model models/oie_model/epoch=14_eval_acc=0.551_v0.ckpt --conj_model models/conj_model/epoch=28_eval_acc=0.854.ckpt --rescore_model models/rescore_model --num_extractions 5 

# cd ..
conda activate sentence_transformers
#conda init bash

# #Form data frame
# python triples_to_df.py \
# --input_query_triples data/triples/$DATASET/queries.txt \
# --input_passage_triples data/triples/$DATASET/passages.txt \
# --rels_file data/triples/$DATASET/qrels.txt \
# --out_file ../data/$DATASET/triples.json \

# cd ..

# #Generate scores
# python generate_triple_scores.py --dataset $DATASET
