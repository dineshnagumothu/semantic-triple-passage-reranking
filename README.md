# Semantic Triple Assisted Learning For Question Answering Passage Re-ranking

## Abstract

Passage re-ranking in question answering (QA) systems is a method to reorder a set of retrieved passages, related to a given question so that answer-containing passages are ranked higher than non-answer-containing passages. With recent advances in language models, passage ranking has become more effective due to improved natural language understanding of the relationship between questions and answer passages. With neural network models, question-passage pairs are used to train a cross-encoder that predicts the semantic relevance score of the pairs and is subsequently used to rank retrieved passages. This paper reports on the use of open information extraction (OpenIE) triples in the form <subject, verb, object> for questions and passages to enhance answer passage ranking in neural network models. Coverage and overlap scores of question-passage triples are studied and a novel loss function is developed using the proposed triple-based features to better learn a cross-encoder model to rerank passages. Experiments on three benchmark datasets are compared to the baseline BERT and ERNIE models using the proposed loss function demonstrating improved passage re-ranking performance.



## Instructions to run the code

<ol>
  <li><b>Create environments</b></li>

  This step creates separate environments for OpenIE6 and sentence-transformers. (Single environment setup will not work because of conflicting packages)
  
  ```
  sh setup.sh
  ```

  <li><b>Extract Triples and Generate Coverage/Overlap Scores</b></li>

  This step prepares the data, extracts OpenIE6 triples from questions and passages and computes the coverage/overlap scores and saves them to disk. 

  You can edit the settings in the bash file to define the number of GPUs ```NUM_GPUS``` to use, positive to negative ratio ```POS_NEG_RATIO```, and number of samples ```MAX_SAMPLES``` to process
  
  ```
  sh feature_generation.sh
  ```

  <li><b>Train and Evaluate the Passage Re-ranking model</b></li>

  This step trains and measures the performance of the re-ranking model with cross-encoder architecture

  Feel free to experiment with different language models, and change ```MODEL_NAME``` to any HuggingFace encoder models. This work experiments with BERT-base ```bert-base-uncased``` and ERNIE-base ```nghuyong/ernie-2.0-base-en```.  You can also select ```FEATURES``` as ```coverage``` or ```overlap``` to alter the loss function using the specified scores
  
  ```
  sh train.sh
  ```
</ol>

Currently, this repo supports the MS MARCO dataset. Support to QUASAR-T and TREC 2019 DL Track is coming soon.

## Cite
Cite our work as 

    @InProceedings{10.1007/978-3-031-41682-8_16,
    author="Nagumothu, Dinesh
    and Ofoghi, Bahadorreza
    and Eklund, Peter W.",
    editor="Fink, Gernot A.
    and Jain, Rajiv
    and Kise, Koichi
    and Zanibbi, Richard",
    title="Semantic Triple-Assisted Learning for Question Answering Passage Re-ranking",
    booktitle="Document Analysis and Recognition - ICDAR 2023",
    year="2023",
    publisher="Springer Nature Switzerland",
    address="Cham",
    pages="249--264",
    isbn="978-3-031-41682-8"
    }

## Contact
For any issues or queries, feel free to contact us at dnagumot@deakin.edu.au

