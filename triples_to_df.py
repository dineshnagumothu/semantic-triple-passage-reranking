import re
import json
import pandas as pd
import argparse
import sys
from pprint import pprint
import nltk
from tqdm import tqdm
import os
from generate_triple_scores import calculate_overlap_scores


from nltk.corpus import stopwords
cstopwords = stopwords.words("english")

def check_passage_sentences(sentences):
    new_sentences = []
    
    for sentence in sentences:
        words = [word for word in sentence.split() if word not in cstopwords]
        repeating = [item for item in set(words) if words.count(item) > 8]
        words_2 = sentence.split('-')
        repeating_2 = [item for item in set(words_2) if words_2.count(item) > 3]
        if(len(repeating)==0 and len(repeating_2)==0):
            new_sentences.append(sentence)
    return new_sentences 

def clean_string(mystring):
    mystring = re.sub(r"\([^()]*\)", "", mystring)
    return re.sub('[^A-Za-z0-9]+', ' ', mystring)
    #return ''.join(e for e in mystring if e.isalnum())

class Triple:
    def __init__(self, subj, rel, obj, text=None, confidence=None):
        self.subj = subj
        self.rel = rel
        self.obj = obj
        self.text = text
        self.confidence = confidence
  
    def __str__(self):
        x=  "Text:"+self.text+"\n"
        x+=  "Confidence:"+str(self.confidence)+"\n"
        x+=  self.subj+" --- "+ self.rel+"--- "+self.obj+"\n---\n"
        return str(x)

def ie6_format(filepath):
    all_triples=[]
    seperator_flag = False
    with open(filepath, "r", encoding="utf-8") as file:
        context_triples = []
        context=""
        for line in file:
            if(line=='\n' or len(line)<=4):
                if(seperator_flag == True):
                  seperator_flag=False;
                  all_triples.append(context_triples)
                  context_triples=[]
            elif ("Dinesh Nagumothu is a PhD student at Deakin University" in line):
                seperator_flag = True
            elif (seperator_flag==True):
                continue
            elif (line[4]==':' and line[:4].replace('.','',1).isdigit()):
                confidence = float(line[:4])       
                triple = line[6:].split(';')
                if(len(triple)<2):
                    continue
                try:
                    triple_text = triple
                    triple[0]=clean_string(triple[0][1:])
                    triple[2]=clean_string(triple[2][:-2])
                    triple[1]=clean_string(triple[1])
                    triple_object = [triple[0], triple[1], triple[2]]

                except:
                    print(triple_text)    
                
                #triple_object = Triple(triple[0], triple[1], triple[2], text=context, confidence=confidence)
                context_triples.append(triple_object)
            else:
                context = line
                #print ("Context")
    print ("Number of items:", len(all_triples))
    
    no_triple_passage_count=0
    for context_triples in all_triples:
        if(len(context_triples)==0):
                no_triple_passage_count+=1
    print ("Number of items without any triples:", no_triple_passage_count)
    '''
    sub_triples = all_triples[:5]
    for triples in sub_triples:
      for triple in triples:
        print (triple)
      print ("----Document End----")
    '''
    return (all_triples)

    ###Formatting triples generated from IE6 Triples
    all_triples=[]
    seperator_flag = False
    with open(filepath, "r", encoding="utf-8") as file:
        context_triples = []
        context=""
        for line in file:
            if(line=='\n' or len(line)<=4):
                if(seperator_flag == True):
                  seperator_flag=False;
                  all_triples.append(context_triples)
                  context_triples=[]
            elif ("Dinesh Nagumothu is a PhD student at Deakin University" in line):
                seperator_flag = True
            elif (seperator_flag==True):
                continue
            elif (line[4]==':' and line[:4].replace('.','',1).isdigit()):
                confidence = float(line[:4])       
                triple = line[6:].split(';')
                if(len(triple)<2):
                    continue
                triple[0]=clean_string(triple[0][1:])
                triple[2]=clean_string(triple[2][:-2])
                triple[1]=clean_string(triple[1])
                #triple_object = Triple(triple[0], triple[1], triple[2], text=context, confidence=confidence)
                triple_object = [triple[0], triple[1], triple[2]]
                context_triples.append(triple_object)
            else:
                context = line
                #print ("Context")
    print ("Number of questions:", len(all_triples))
    
    return (all_triples)  

def convert_triple_to_list(all_triple_objects):
  all_triples = []
  all_confidences = []
  for triple_objects in all_triple_objects:
    context_triples = []
    context_confidences = []
    for triple_object in triple_objects:
      context_triples.append([triple_object.subj, triple_object.rel, triple_object.obj])
      context_confidences.append(triple_object.confidence)
    all_triples.append(context_triples)
    all_confidences.append(context_confidences)
  return all_triples, all_confidences

def convert_to_sentences(passages):
  all_sentences=[]
  sentence_counter = 0
  for i in range(len(passages)):
    passage_sentences = nltk.sent_tokenize(passages[i])
    passage_sentences = check_passage_sentences(passage_sentences)
    all_sentences.append(passage_sentences)
    sentence_counter += len(passage_sentences)
  print ("Total number of Documents", len(all_sentences))
  print ("Total number of Sentences with seperator sentence", sentence_counter)
  return all_sentences

def read_rels_file(filename):
    print ("Reading rels from "+filename)
    qids = []
    pids = []
    rels = []
    with open(filename, 'r') as f:
      for line in f:
        qid, pid, rel = line.strip().split("\t")
        qids.append(qid)
        pids.append(pid)
        rels.append(rel)
    print ("Number of samples :"+str(len(qids)))
    return qids, pids, rels

def read_lines_from_text(filename):
  print ("Reading from "+filename)
  lines = []
  with open(filename, 'r') as f:
      for line in f:
        lines.append(line.strip())
  print ("Number of lines :"+str(len(lines)))
  return lines

def id_text_triple_dict(ids, texts, triples):
    dictionary = {}
    for i in range(len(ids)):
        dictionary[ids[i]]= {"text":texts[i], "triples":triples[i]}
    return dictionary

if __name__=="__main__":
    my_parser = argparse.ArgumentParser(description='Convert Triples to Dataframe for parsing')
    my_parser.add_argument('--dataset', help='Name of the dataset', required=True)
    args = my_parser.parse_args()

    dataset = args.dataset
    data_folder = "data/"+dataset
    openie6_data_folder = "openie6/data/"+dataset

    qid_file = os.path.join(data_folder, 'train_qids.txt')
    pid_file = os.path.join(data_folder, 'train_pids.txt')
    queries_file = os.path.join(data_folder, 'train_queries.txt')
    passages_file = os.path.join(data_folder, 'train_passages.txt')
    qids = read_lines_from_text(qid_file)
    pids = read_lines_from_text(pid_file)
    queries = read_lines_from_text(queries_file)
    passages = read_lines_from_text(passages_file)

    query_triples_file = os.path.join(openie6_data_folder, 'queries_triples.txt')
    ie6_query_triples = ie6_format(query_triples_file)

    passages_triples_file = os.path.join(openie6_data_folder, 'passages_triples.txt')
    ie6_passages_triples = ie6_format(passages_triples_file)

    try:
        assert(len(pids)==len(passages))
        assert(len(pids)==len(ie6_passages_triples))
        assert(len(qids)==len(queries))
        assert(len(qids)==len(ie6_query_triples))
    except AssertionError:
        print ("The files are corrupt; Lengths of ids file and text file did not match")
        sys.exit()

    queries_dict = id_text_triple_dict(qids, queries, ie6_query_triples)
    passages_dict = id_text_triple_dict(pids, passages, ie6_passages_triples) 

    rels_file = os.path.join(data_folder, 'qid_pid_rels.tsv')
    qid_rels, pid_rels, rels = read_rels_file(rels_file)
    
    df_queries = []
    df_passages = []
    df_query_triples = []
    df_passages_triples = []
    for i in range(len(qid_rels)):
        df_queries.append(queries_dict[qid_rels[i]]['text'])
        df_passages.append(passages_dict[pid_rels[i]]['text'])
        df_query_triples.append(queries_dict[qid_rels[i]]['triples'])
        df_passages_triples.append(passages_dict[pid_rels[i]]['triples'])

    out_df = pd.DataFrame()
    out_df['qid'] = qid_rels
    out_df['query'] = df_queries
    out_df['query_triples'] = df_query_triples
    out_df['pid'] = pid_rels
    out_df['passage'] = df_passages
    out_df['passage_triples'] = df_passages_triples
    out_df['relevance'] = rels

    out_df = calculate_overlap_scores(out_df)

    out_file = os.path.join(data_folder, 'train_df.json')
    out_df.to_json(out_file, orient='records', lines=True)

    positive = out_df[out_df['relevance']=='1']
    print (positive['query'].iloc[0])
    print (positive['query_triples'].iloc[0])
    print (positive['passage'].iloc[0])
    print (positive['passage_triples'].iloc[0])

    print (positive['subject_overlap_score'].iloc[0])
    print (positive['predicate_overlap_score'].iloc[0])
    print (positive['object_overlap_score'].iloc[0])
    print (positive['overall_overlap_score'].iloc[0])

    print (positive['subject_coverage_score'].iloc[0])
    print (positive['predicate_coverage_score'].iloc[0])
    print (positive['object_coverage_score'].iloc[0])
    print (positive['overall_coverage_score'].iloc[0])

    print (positive.head(1))
    print (queries_dict['1001876'])