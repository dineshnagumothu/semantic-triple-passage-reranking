import nltk
import json
import argparse
import os

def convert_to_sentences(passages):
  all_sentences=[]
  sentence_counter = 0
  actual_sentence_counter = 0
  for i in range(len(passages)):
    passage_sentences = nltk.sent_tokenize(passages[i])
    actual_sentence_counter += len(passage_sentences)
    seperator_sentence = "Dinesh Nagumothu is a PhD student at Deakin University - " + str(i) + "."
    passage_sentences.append(seperator_sentence)
    all_sentences.extend(passage_sentences)
    sentence_counter += len(passage_sentences)
  #print ("Total number of Documents", len(all_sentences))
  #print ("Total number of Sentences", actual_sentence_counter)
  #print ("Total number of Sentences with seperator sentence", sentence_counter)
  return all_sentences


def write_sentences_to_file(sentences, out_file_name):
  print ("Writing Sentences to " + out_file_name)
  with open(out_file_name, 'w') as f:
    for sentence in sentences:
      f.write("%s\n" % sentence)

def read_passages_from_file(in_file_name):
    #print ("Reading Passages from "+in_file_name)
    passages = []
    with open(in_file_name, 'r') as f:
        for line in f:
            passages.append(line.strip())
    #print ("Number of passages :"+str(len(passages)))
    return passages

if __name__=="__main__":
  my_parser = argparse.ArgumentParser(description='Convert Passages to Sentences')
  my_parser.add_argument('--dataset', help='Input Dataset Name', required=True)

  # my_parser.add_argument('--input_passages_file', help='Input Passages File Name', required=True)
  # my_parser.add_argument('--input_queries_file', help='Input Queries File Name', required=True)
  # my_parser.add_argument('--output_passages_file', help='Output Passages File Name', required=True)
  # my_parser.add_argument('--output_queries_file', help='Output Queries File Name', required=True)

  args = my_parser.parse_args()
  dataset = args.dataset
  dataset = args.dataset
  data_folder = 'data/'+dataset+'/'
  input_queries_file = os.path.join(data_folder, 'train_queries.txt')
  input_passages_file = os.path.join(data_folder, 'train_passages.txt')

  passages = read_passages_from_file(input_passages_file)
  queries = read_passages_from_file(input_queries_file)
  print ("Total number of passages in the corpus ", len(passages))
  print ("Total number of questions in the corpus ", len(queries))

  all_passage_sentences = convert_to_sentences(passages)
  all_query_sentences = convert_to_sentences(queries)

  openie6_data_folder = 'openie6/data/'
  output_folder = os.path.join(openie6_data_folder, dataset)

  if (os.path.isdir(output_folder)==False):
      os.mkdir(output_folder)
  
  query_sents_file = os.path.join(output_folder, 'queries_sentences.txt')
  passage_sents_file = os.path.join(output_folder, 'passages_sentences.txt')
  write_sentences_to_file(all_query_sentences, query_sents_file)
  write_sentences_to_file(all_passage_sentences, passage_sents_file)