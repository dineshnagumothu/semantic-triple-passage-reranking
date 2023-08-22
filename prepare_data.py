import argparse
import logging
from datetime import datetime
import gzip
import os
import tarfile
import tqdm
import sys
from sentence_transformers import util


def check_and_download_file(filename, url):
    if not os.path.exists(filename):
        logging.info("Downloading data from..."+ str(url))
        util.http_get(url, filename)

def read_msmarco_qrels_files(path, max_samples=1000000, pos_neg_ratio=4):
    qrels = set()
    qid_num_count = {}
    with gzip.open(path, 'rt') as fIn:
        for line in tqdm.tqdm(fIn, unit_scale=True):
            qid, pos_id, neg_id = line.strip("\t").split()
            pos_text = qid+"\t"+pos_id+"\t1"
            neg_text = qid+"\t"+neg_id+"\t0"

            if qid not in qid_num_count:
                qid_num_count[qid]=[0,0]
            
            pos_count = qid_num_count[qid][0]
            neg_count = qid_num_count[qid][1]
                
            if pos_count>1:
                continue
            if neg_count>pos_neg_ratio:
                continue
            qrels.add(pos_text)
            qid_num_count[qid][0] = pos_count+1
            qrels.add(neg_text)
            qid_num_count[qid][1] = neg_count+1
            if len(qrels)>max_samples:
                break
    qrels = list(qrels)
    return qrels  

def read_msmarco_files(path):
    dictionary = {}
    with open(path, 'r', encoding='utf8') as fIn:
        for line in tqdm.tqdm(fIn, unit_scale=True):
            id, text = line.strip().split("\t")
            dictionary[id] = text
    return dictionary

def download_data(dataset, data_folder):
    if dataset == 'msmarco':
        ### Download and extract train collections
        collection_filepath = os.path.join(data_folder, 'collection.tsv')
        if not os.path.exists(collection_filepath):
            tar_filepath = os.path.join(data_folder, 'collection.tar.gz')
            check_and_download_file(tar_filepath, 'https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz')   
            with tarfile.open(tar_filepath, "r:gz") as tar:
                tar.extractall(path=data_folder)

        ### Download and extract train queries
        queries_filepath = os.path.join(data_folder, 'queries.train.tsv')
        if not os.path.exists(queries_filepath):
            tar_filepath = os.path.join(data_folder, 'queries.tar.gz')
            check_and_download_file(tar_filepath, 'https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz')
            with tarfile.open(tar_filepath, "r:gz") as tar:
                tar.extractall(path=data_folder)
        
        # Download and extract the training file            
        train_filepath = os.path.join(data_folder, 'qidpidtriples.train.full.2.tsv.gz')        
        check_and_download_file(train_filepath,'https://msmarco.blob.core.windows.net/msmarcoranking/qidpidtriples.train.full.2.tsv.gz')

    else:
        print("Incorrect dataset name")
        sys.exit(1)    

def write_list(path, text_list):
    with open(path, 'w', encoding='utf8') as out:
        for item in text_list:
            out.write(item+"\n")

def save_query_passage_files(dataset, data_folder, max_samples, pos_neg_ratio):
    #Save query file and qid to txt files using max_samples
    #Save collection file and pid to txt files using max_samples
    queries = []
    qids = []
    passages = []
    pids = []
    qid_pid_rels = []
    if dataset=='msmarco':
        train_filepath = os.path.join(data_folder, 'qidpidtriples.train.full.2.tsv.gz')
        qid_pid_rels = read_msmarco_qrels_files(train_filepath, max_samples=max_samples, pos_neg_ratio=pos_neg_ratio)
        write_list(os.path.join(data_folder,'qid_pid_rels.tsv'), qid_pid_rels)

        for item in qid_pid_rels:
            qid, pid, rel = item.strip("\t").split()
            if qid not in qids:
                qids.append(qid)
            if pid not in pids:
                pids.append(pid)

        queries_filepath = os.path.join(data_folder, 'queries.train.tsv')
        queries_dict = read_msmarco_files(queries_filepath)
        for qid in qids:
            queries.append(queries_dict[qid])
        write_list(os.path.join(data_folder,'train_qids.txt'), qids)
        write_list(os.path.join(data_folder,'train_queries.txt'), queries)

        collection_filepath = os.path.join(data_folder, 'collection.tsv')
        passages_dict = read_msmarco_files(collection_filepath)
        for pid in pids:
            passages.append(passages_dict[pid])
        write_list(os.path.join(data_folder,'train_pids.txt'), pids)
        write_list(os.path.join(data_folder,'train_passages.txt'), passages)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Train MSMARCO dataset')
    parser.add_argument('--dataset', type=str, default='msmarco', help='Dataset to train on')
    parser.add_argument('--max_samples', type=int, default='1000000', help='Number of pairs to train on')
    parser.add_argument('--pos_neg_ratio', type=int, default='4', help='Number of negative passages to one positive passage')
    parser.add_argument('--random_seed', type=int, default='2020', help='Random seed value')

    args = parser.parse_args()
    dataset = args.dataset
    data_folder = 'data/'+dataset+'/'
    max_samples = args.max_samples
    pos_neg_ratio = args.pos_neg_ratio
    download_data(dataset, data_folder)
    save_query_passage_files(dataset, data_folder, max_samples, pos_neg_ratio)