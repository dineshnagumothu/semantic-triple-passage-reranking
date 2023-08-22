"""
This examples show how to train a Cross-Encoder for the MS Marco dataset (https://github.com/microsoft/MSMARCO-Passage-Ranking).

The query and the passage are passed simoultanously to a Transformer network. The network then returns
a score between 0 and 1 how relevant the passage is for a given query.

The resulting Cross-Encoder can then be used for passage re-ranking: You retrieve for example 100 passages
for a given query, for example with ElasticSearch, and pass the query+retrieved_passage to the CrossEncoder
for scoring. You sort the results then according to the output of the CrossEncoder.

This gives a significant boost compared to out-of-the-box ElasticSearch / BM25 ranking.

Running this script:
python train_cross-encoder.py
"""
import argparse
import sys

#sys.path.insert(0, '/home/dnagumot/dnagumot/passage_ranking/sentence-transformers')
#sys.path.insert(0, '/home/dnagumot/dnagumot/passage_ranking/transformers-main/src')
#print (sys.path)

from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from sentence_transformers import InputExample
import logging
from datetime import datetime
import gzip
import os
import tarfile
import tqdm
import pandas as pd
import random
import json

from evaluate import mrr_recall
import numpy as np

from ms_marco_eval import compute_metrics_from_files


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


#mrr_recall('trained_output_msmarco_2.json', False, False, ctx_col='new_ranked_contexts')
#sys.exit()

#First, we define the transformer model we want to fine-tune
#model_name = 'distilroberta-base'

my_parser = argparse.ArgumentParser(description='Train MSMARCO dataset')
my_parser.add_argument('--dataset', type=str, default='msmarco', help='Dataset to train on')
my_parser.add_argument('--model_name', help='Model Name', required=True)
my_parser.add_argument('--run', help='Run number', required=True)
my_parser.add_argument('--alpha', type=float, help='Alpha value', required=True)
my_parser.add_argument('--input_features', type=str, default='coverage', help='Input scores')
my_parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
my_parser.add_argument('--epochs', type=int, default=2, help='Number of Epochs to Train')

args = my_parser.parse_args()


model_name = args.model_name
run=args.run
alpha=args.alpha
dataset = args.dataset

if dataset == 'msmarco':
    if model_name=='bert-base-uncased':
        model_dir_name = 'msmarco_bert_base/'
    elif model_name=='bert-large-uncased-whole-word-masking':
        model_dir_name = 'msmarco_bert_large/'
    elif model_name=='nghuyong/ernie-2.0-base-en':
        model_dir_name = 'msmarco_ernie_base/'
    elif model_name=='nghuyong/ernie-2.0-large-en':
        model_dir_name = 'msmarco_ernie_large/'
elif dataset == 'quasart':
    if model_name=='bert-base-uncased':
        model_dir_name = 'quasart_bert_base/'
    elif model_name=='bert-large-uncased-whole-word-masking':
        model_dir_name = 'quasart_bert_large/'
    elif model_name=='nghuyong/ernie-2.0-base-en':
        model_dir_name = 'quasart_ernie_base/'
    elif model_name=='nghuyong/ernie-2.0-large-en':
        model_dir_name = 'quasart_ernie_large/'
    

train_batch_size = args.batch_size    
num_epochs = args.epochs
if(args.input_features!='coverage'):
    model_save_path = 'output/'+model_dir_name+model_name.replace("/", "-")+args.input_features+'-alpha'+str(alpha).replace('.','')+'-run'+str(run)
else:
    model_save_path = 'output/'+model_dir_name+model_name.replace("/", "-")+'-alpha'+str(alpha).replace('.','')+'-run'+str(run)
    



# We train the network with as a binary label task
# Given [query, passage] is the label 0 = irrelevant or 1 = relevant?
# We use a positive-to-negative ratio: For 1 positive sample (label 1) we include 4 negative samples (label 0)
# in our training setup. For the negative samples, we use the triplets provided by MS Marco that
# specify (query, positive sample, negative sample).
pos_neg_ration = 4

# Maximal number of training samples we want to use
max_train_samples = 2e4

#We set num_labels=1, which predicts a continous score between 0 and 1
random.seed(100)
#input_features = []

if (args.input_features == 'coverage'):
    input_features = ['coverage_score']
    len_input_features = 4
elif (args.input_features =='overlap'):
    input_features = ['overlap_score']
    len_input_features = 4

output_prediction_file = 'predictions/'+model_dir_name+args.input_features+'_'+model_name.replace("/", "-")+'_alpha'+str(alpha).replace('.','')+'_run'+str(run)+'.json'

output_rank_file = 'predictions/'+model_dir_name+args.input_features+'_'+model_name.replace("/", "-")+'_alpha'+str(alpha).replace('.','')+'_run'+str(run)+'.tsv'

num_labels = 1
late_fusion=False
model = CrossEncoder(model_name, num_labels=num_labels, max_length=512, len_features = len_input_features, late_fusion=late_fusion)

if late_fusion==True:
    model_save_path+="_latefusion"
    output_prediction_file = 'predictions/'+model_dir_name+args.input_features+'_'+model_name.replace("/", "-")+'_alpha'+str(alpha).replace('.','')+'_run'+str(run)+'_latefusion.json'

    output_rank_file = 'predictions/'+model_dir_name+args.input_features+'_'+model_name.replace("/", "-")+'_alpha'+str(alpha).replace('.','')+'_run'+str(run)+'_latefusion.tsv'


#We set num_labels=1, which predicts a continous score between 0 and 1
#model = CrossEncoder(model_name, num_labels=1, max_length=512)


### Now we read the MS Marco dataset
if dataset == 'msmarco':
    data_folder = 'msmarco-data'
elif dataset == 'quasart':
    data_folder = 'quasart'

    
os.makedirs(data_folder, exist_ok=True)

###print information###
print("-"*30+"Run info"+"-"*30)
if len(input_features)==0:
    print("Input - Text only")
else:
    t  = "Input - Text + "
    for in_f in input_features:
        t+=in_f+" + "
    t=t[:-3]
    print (t)
print ("Number of labels - " + str(num_labels))
print ("Late Fusion Settings Enabled - " +str(late_fusion))
print ("Data Folder - " + data_folder)
print ("Output Prediction File - " + output_prediction_file)
print ("Output Rank File - " + output_rank_file)
print ("Number of Input Features if any - " + str(len_input_features))
print ("Batch size - " + str(train_batch_size))
print ("Model path - " + model_save_path)
print("-"*30+"Run info end"+"-"*30)
###print information###

'''
#### Read the corpus files, that contain all the passages. Store them in the corpus dict
corpus = {}
collection_filepath = os.path.join(data_folder, 'collection.tsv')
if not os.path.exists(collection_filepath):
    tar_filepath = os.path.join(data_folder, 'collection.tar.gz')
    if not os.path.exists(tar_filepath):
        logging.info("Download collection.tar.gz")
        util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz', tar_filepath)

    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)

with open(collection_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        corpus[pid] = passage


### Read the train queries, store in queries dict
queries = {}
queries_filepath = os.path.join(data_folder, 'queries.train.tsv')
if not os.path.exists(queries_filepath):
    tar_filepath = os.path.join(data_folder, 'queries.tar.gz')
    if not os.path.exists(tar_filepath):
        logging.info("Download queries.tar.gz")
        util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz', tar_filepath)

    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)


with open(queries_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        queries[qid] = query



i=0
with open('rel_file.tsv', 'w') as outfile:
    relevant_pairs = []
    print ("Reading rel_file")
    dev_qrels_filepath = os.path.join(data_folder, 'qrels.dev.tsv')
    with open(dev_qrels_filepath, 'r', encoding='utf8') as fIn:
        for line in fIn:
            qid, _, pid, _ = line.strip().split('\t')
            rel_item = str(qid)+"\t"+str(pid)
            if(i<5):
                print (rel_item)
            if(rel_item not in relevant_pairs):
                relevant_pairs.append(rel_item)
                outfile.write(rel_item)
                outfile.write("\n")
            i+=1

with open('rank_file.tsv', 'w') as outfile:
    qid_rank = {}
    eval_filepath = os.path.join(data_folder, 'top1000.dev.tsv')
    print ("Reading Eval file")
    with open(eval_filepath, 'r', encoding='utf8') as fIn:
        for line in tqdm.tqdm(fIn):
            qid, pid, query, passage = line.strip().split("\t")
            if qid not in qid_rank.keys():
                qid_rank[qid] = 1
            else:
                qid_rank[qid] = qid_rank[qid]+1
            rank = str(qid_rank[qid])
            write_item = str(qid)+"\t"+str(pid)+"\t"+rank
            outfile.write(write_item)
            outfile.write("\n")
sys.exit()
'''

###Download and evaluate the eval set

def read_eval_input(eval_filepath, tar_filepath, dev_qrels_filepath):
    if not os.path.exists(eval_filepath):
        if not os.path.exists(tar_filepath):
            logging.info("Download top1000.dev.tar.gz")
            util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/top1000.dev.tar.gz', tar_filepath)

        with tarfile.open(tar_filepath, "r:gz") as tar:
            tar.extractall(path=data_folder)

    relevant_pairs = []
    with open(dev_qrels_filepath, 'r', encoding='utf8') as fIn:
        for line in fIn:
            qid, _, pid, _ = line.strip().split('\t')
            rel_item = '\t'.join([qid, pid])
            if(rel_item not in relevant_pairs):
                relevant_pairs.append(rel_item)
    i=0
    test_output = {}
    qid_rank = {}
    with open(eval_filepath, 'r', encoding='utf8') as fIn:
        for line in tqdm.tqdm(fIn):
            qid, pid, query, passage = line.strip().split("\t")
            if qid not in test_output.keys():
                test_output[qid] = {'qid': qid, 'question':query, 'ranked_contexts':[]}
                qid_rank[qid] = 1
            else:
                qid_rank[qid] = qid_rank[qid] + 1
            rank = qid_rank[qid]
            query_passage_pair = '\t'.join([qid, pid])
            label=0
            if query_passage_pair in relevant_pairs:
                label=1
            test_output[qid]['ranked_contexts'].append({'string':passage, 'pid':pid, 'rank':rank, 'answer_contain': int(label)})
            i+=1
    return test_output

def read_eval_from_file(filename):
    print ("Reading Eval from File")
    test_output={}
    with open(filename, 'r') as fIn:
        for line in tqdm.tqdm(fIn):
            json_obj = json.loads(line)
            qid = json_obj['qid']
            test_output[qid] = json_obj
    return test_output


if dataset=='msmarco':
    eval_filepath = os.path.join(data_folder, 'top1000.dev.tsv')        
    tar_filepath = os.path.join(data_folder, 'top1000.dev.tar.gz')
    dev_qrels_filepath = os.path.join(data_folder, 'qrels.dev.tsv')
    
    if not os.path.exists('init_msmarco_output.json'):
        test_output = read_eval_input(eval_filepath, tar_filepath, dev_qrels_filepath)
        with open('init_msmarco_output.json', 'w') as outfile:
            for entry in test_output.keys():
                json.dump(test_output[entry], outfile)
                outfile.write('\n')
    else:
        test_output = read_eval_from_file('init_msmarco_output.json')
    print ("Total number of Dev Questions: "+ str(len(test_output.keys())))
    
    '''
   
    '''
    print ("Performing initial recall")
    mrr_recall('init_msmarco_output.json', False, False)

#sys.exit()
elif dataset=='quasart':
    test_data_df = pd.read_json("quasart_data/model_input/quasart_test_overlap.json")
    print (test_data_df.head())
    qids = list(test_data_df['qid'].unique())
    print ("Total number of Test questions: "+ str(len(qids)))
    
    test_output = {}

    for i in tqdm.tqdm(range(len(test_data_df))):
        qid = test_data_df['qid'].iloc[i]
        context = test_data_df['context'].iloc[i]
        answer_contain = test_data_df['answer_contain'].iloc[i]

        coverage_scores = []
        if('coverage_score' in input_features):
            #subject_coverage_score = test_data_df['subject_coverage_score'].iloc[i]
            #predicate_coverage_score = test_data_df['predicate_coverage_score'].iloc[i]
            #object_coverage_score = test_data_df['object_coverage_score'].iloc[i]
            #overall_coverage_score = test_data_df['overall_coverage_score'].iloc[i]

            subject_coverage_score = round(test_data_df['subject_coverage_score'].iloc[i]*100,2)
            predicate_coverage_score = round(test_data_df['predicate_coverage_score'].iloc[i]*100,2)
            object_coverage_score = round(test_data_df['object_coverage_score'].iloc[i]*100,2)
            overall_coverage_score = round(test_data_df['overall_coverage_score'].iloc[i]*100,2)


            coverage_scores.extend([subject_coverage_score, predicate_coverage_score, object_coverage_score, overall_coverage_score])

        if('overlap_score' in input_features):
            #subject_overlap_score = test_data_df['subject_overlap_score'].iloc[i]
            #predicate_overlap_score = test_data_df['predicate_overlap_score'].iloc[i]
            #object_overlap_score = test_data_df['object_overlap_score'].iloc[i]
            #overall_overlap_score = test_data_df['overall_overlap_score'].iloc[i]

            subject_overlap_score = round(test_data_df['subject_overlap_score'].iloc[i]*100,2)
            predicate_overlap_score = round(test_data_df['predicate_overlap_score'].iloc[i]*100,2)
            object_overlap_score = round(test_data_df['object_overlap_score'].iloc[i]*100,2)
            overall_overlap_score = round(test_data_df['overall_overlap_score'].iloc[i]*100,2)

            coverage_scores.extend([subject_overlap_score, predicate_overlap_score, object_overlap_score, overall_overlap_score])

        if qid not in test_output.keys():
            question = test_data_df['question'].iloc[i]
            #answer = test_data_df['answer'].iloc[i]
            rank=1
            test_output[qid] = {'question':question, 'ranked_contexts':[]}
        else:
            rank+=1
        test_output[qid]['ranked_contexts'].append({'string':context,'rank':rank, 'answer_contain': int(answer_contain), 'overlap_scores':coverage_scores}) 

    
    with open('init_output.json', 'w') as outfile:
        for entry in test_output.keys():
            json.dump(test_output[entry], outfile)
            outfile.write('\n')
    
    
    mrr_recall('init_output.json', False, False)



##My code
train_samples = []
dev_samples = []

if dataset=='msmarco':
    #train_data_df = pd.read_json("msmarco_output/msmarco_train_df_coverage.json",lines=True)
    train_data_df = pd.read_json("msmarco_output/msmarco_train_df_coverage_all.json",lines=True)
    df_qids = list(train_data_df['qid'].unique())
    dev_qids = random.sample(df_qids, 5000)
elif dataset=='quasart':
    train_data_df = pd.read_json("quasart_data/model_input/quasart_dev_overlap.json")
    df_qids = list(train_data_df['qid'].unique())
    dev_qids = random.sample(df_qids, 100)

print ("Total number of questions: "+ str(len(df_qids)))
print ("Total number of dev questions: "+ str(len(dev_qids)))

dev_temp_samples = {}

cnt=0

qids=[]
pos_position=0
neg_position=0

num_max_negatives=200
num_max_positives=100


qids_pos_neg_count = {}

col_names = {'msmarco':['passage', 'query'], 'quasart':['context', 'question']}

for i in tqdm.tqdm(range(len(train_data_df))):
    qid = train_data_df['qid'].iloc[i]
    context = train_data_df[col_names[dataset][0]].iloc[i]
    question = train_data_df[col_names[dataset][1]].iloc[i]
    answer_contain = train_data_df['answer_contain'].iloc[i]
    
    coverage_scores = []
    if('coverage_score' in input_features):
        #subject_coverage_score = train_data_df['subject_coverage_score'].iloc[i]
        #predicate_coverage_score = train_data_df['predicate_coverage_score'].iloc[i]
        #object_coverage_score = train_data_df['object_coverage_score'].iloc[i]
        #overall_coverage_score = train_data_df['overall_coverage_score'].iloc[i]

        subject_coverage_score = round(train_data_df['subject_coverage_score'].iloc[i]*100,2)
        predicate_coverage_score = round(train_data_df['predicate_coverage_score'].iloc[i]*100,2)
        object_coverage_score = round(train_data_df['object_coverage_score'].iloc[i]*100,2)
        overall_coverage_score = round(train_data_df['overall_coverage_score'].iloc[i]*100,2)

        coverage_scores.extend([subject_coverage_score, predicate_coverage_score, object_coverage_score, overall_coverage_score])

    if('overlap_score' in input_features):
        #subject_ovelap_score = train_data_df['subject_overlap_score'].iloc[i]
        #predicate_overlap_score = train_data_df['predicate_overlap_score'].iloc[i]
        #object_overlap_score = train_data_df['object_overlap_score'].iloc[i]
        #overall_overlap_score = train_data_df['overall_overlap_score'].iloc[i]

        subject_overlap_score = round(train_data_df['subject_overlap_score'].iloc[i]*100,2)
        predicate_overlap_score = round(train_data_df['predicate_overlap_score'].iloc[i]*100,2)
        object_overlap_score = round(train_data_df['object_overlap_score'].iloc[i]*100,2)
        overall_overlap_score = round(train_data_df['overall_overlap_score'].iloc[i]*100,2)

        coverage_scores.extend([subject_overlap_score, predicate_overlap_score, object_overlap_score, overall_overlap_score])
        
        
    if qid in dev_qids:
        if qid not in dev_temp_samples.keys():
            temp_sample = {'query': question,'positive':[], 'negative':[], 'pos_coverage_scores':[],'neg_coverage_scores':[]}
            if(answer_contain==1):
                temp_sample['positive'].append(context)
                temp_sample['pos_coverage_scores'].append(coverage_scores)
            else:
                temp_sample['negative'].append(context)
                temp_sample['neg_coverage_scores'].append(coverage_scores)
            dev_temp_samples[qid] = temp_sample
        else:
            if(answer_contain==1):
                dev_temp_samples[qid]['positive'].append(context)
                temp_sample['pos_coverage_scores'].append(coverage_scores)
            else:
                dev_temp_samples[qid]['negative'].append(context)
                temp_sample['neg_coverage_scores'].append(coverage_scores)
    else:
        train_samples.append(InputExample(texts=[question, context], label=answer_contain, overlap_scores = coverage_scores))
        cnt+=1

    if cnt >= max_train_samples:
        break
        
for key, value in dev_temp_samples.items():
    dev_samples.append(value)

#train_samples=train_sampled[:2000]
    
print ("Number of train samples:", len(train_samples))
print ("Number of Dev samples:",len(dev_samples))
#sys.exit()
#Till here my code

# We create a DataLoader to load our train samples
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

# We add an evaluator, which evaluates the performance during training
# It performs a classification task and measures scores like F1 (finding relevant passages) and Average Precision
evaluator = CERerankingEvaluator(dev_samples, name='train-eval')

# Configure the training
warmup_steps = 5000
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=5000,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          alpha=alpha,
          use_amp=True)

#Save latest model
model.save(model_save_path+'-trained')
model = CrossEncoder(model_save_path+"_best_model", max_length=512, len_features = len_input_features, late_fusion=late_fusion)
i=0
##Model predictions
avg_acc = 0.0
avg_loss = 0.0
with open(output_rank_file, 'w') as rank_outfile:
    with open(output_prediction_file, 'w') as outfile:
        for entry in tqdm.tqdm(test_output.keys()): 
            question = test_output[entry]['question']
            if dataset=='msmarco':
                qid = test_output[entry]['qid']
            elif dataset=='quasart':
                qid = str(entry)
            ranked_contexts = test_output[entry]['ranked_contexts']
            #model_input = [[question, doc['string']] for doc in ranked_contexts]
            model_input = []
            if dataset=='msmarco':
                for doc in ranked_contexts:
                    model_input.append({"sentences":[question, doc['string']], "overlap_scores":[0.0,0.0,0.0,0.0]})
            elif dataset=='quasart':
                for doc in ranked_contexts:
                    model_input.append({"sentences":[question, doc['string']], "overlap_scores":doc['overlap_scores']})
            pred_scores = model.predict(model_input, convert_to_numpy=True, show_progress_bar=False)
            pred_scores_argsort = np.argsort(-pred_scores)  #Sort in decreasing order
            acc = 0.0
            loss = 0.0
            for j in range(len(ranked_contexts)):
                actual = ranked_contexts[j]['answer_contain']
                loss += abs(actual-pred_scores[j])
                if pred_scores[j] > 0.5:
                    predicted = 1
                else:
                    predicted = 0
                if actual == predicted:
                    acc += 1.0
            acc = acc / len(ranked_contexts)
            loss = loss / len(ranked_contexts)
            #print ("Accuracy: "+ str(acc))
            #print ("Loss: "+ str(loss))
            avg_acc += acc
            avg_loss += loss

            i+=1
            new_ranked_contexts = []
            rank=1
            for ix in pred_scores_argsort:
                if dataset=='msmarco':
                    pid = ranked_contexts[ix]['pid']
                elif dataset=='quasart':
                    pid = int(ix)
                new_ranked_contexts.append({'string':model_input[ix]['sentences'][1],
                                            'pid':pid, 
                                            'rank':rank, 
                                            'answer_contain':ranked_contexts[ix]['answer_contain']})
                
                write_item = "%s\t%s\t%s" %(qid, pid, rank)
                rank_outfile.write(write_item)
                rank_outfile.write("\n")
                
                rank+=1
            test_output[entry]['new_ranked_contexts'] = new_ranked_contexts
            #print (type(test_output))
            #print (type(test_output[entry]))
            json.dump(test_output[entry], outfile)
            outfile.write('\n')
avg_acc = avg_acc / i
avg_loss = avg_loss / i
print ("Average Accuracy: "+ str(avg_acc))
print ("Average Loss: "+ str(avg_loss))


write_filename = 'output_logs/'+model_dir_name+args.input_features+'_'+model_name.replace("/", "-")+'_alpha'+str(alpha).replace('.','')+'_run'+str(run)+'.txt'

write_text = model_dir_name+args.input_features+'_'+model_name.replace("/", "-")+'_alpha'+str(alpha).replace('.','')+'_run'+str(run)+'.txt'

if late_fusion==True:
    write_filename = 'output_logs/'+model_dir_name+args.input_features+'_'+model_name.replace("/", "-")+'_alpha'+str(alpha).replace('.','')+'_run'+str(run)+'_latefusion.txt'

write_text = model_dir_name+args.input_features+'_'+model_name.replace("/", "-")+'_alpha'+str(alpha).replace('.','')+'_run'+str(run)+'_latefusion.txt'

mrr_recall(output_prediction_file, False, False, ctx_col='new_ranked_contexts', write_filename=write_filename, write_text=write_text)

###MS MARCO Official Eval
if dataset=='msmarco':
    metrics = compute_metrics_from_files("rel_file.tsv", output_rank_file)
    print('#####################')
    for metric in sorted(metrics):
        print('{}: {}'.format(metric, metrics[metric]))
    print('#####################')
        