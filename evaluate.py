from os.path import join
import json
round_dec_points = 3
import re
from tqdm import tqdm

def get_context_pseudo_label(context, answer):
    if re. compile(r'\b({0})\b'. format(answer.strip()), flags=re. IGNORECASE).search(context.strip()):
    #if answer.strip() in context.strip():
        return 1
    else:
        return 0

def mrr_recall(json_file, is_2ndcls, is_3rdcls, ctx_col='ranked_contexts', write_filename=None, write_text=None):
    question_count, answer_passage_ranks = get_answer_passage_ranks(json_file, is_2ndcls, is_3rdcls, ctx_col)
    mr = calculate_mr(answer_passage_ranks)
    mrr_v1 = calculate_mrr_v1(answer_passage_ranks)
    mrr_v2 = calculate_mrr_v2_at(answer_passage_ranks, 10)
    recall_at_1 = calculate_recall_at(answer_passage_ranks, 1)
    recall_at_2 = calculate_recall_at(answer_passage_ranks, 2)
    recall_at_3 = calculate_recall_at(answer_passage_ranks, 3)
    recall_at_4 = calculate_recall_at(answer_passage_ranks, 4)
    recall_at_5 = calculate_recall_at(answer_passage_ranks, 5)
    recall_at_10 = calculate_recall_at(answer_passage_ranks, 10)
    recall_at_20 = calculate_recall_at(answer_passage_ranks, 20)
    recall_at_50 = calculate_recall_at(answer_passage_ranks, 50)

    print('# of questions=', question_count)
    print('MR (Mean Rank)=', mr)
    print('mrr v1 (without null answers)=', mrr_v1)
    print('mrr v2 (with null answers)=', mrr_v2)
    print('recall@1=', recall_at_1)
    print('recall@2=', recall_at_2)
    print('recall@3=', recall_at_3)
    print('recall@4=', recall_at_4)
    print('recall@5=', recall_at_5)
    print('recall@10=', recall_at_10)
    print('recall@20=', recall_at_20)
    print('recall@50=', recall_at_50)
    print('for latex: ' + str("%.3f" % round(mrr_v2, 3)) + ' & ' + str("%.3f" % round(recall_at_1, 3)) + ' & ' + str("%.3f" % round(recall_at_2, 3))
          + ' & ' + str("%.3f" % round(recall_at_3, 3)) + ' & ' + str("%.3f" % round(recall_at_4, 3)) + ' & ' + str("%.3f" % round(recall_at_5, 3))
          + ' & ' + str("%.3f" % round(recall_at_10, 3)))
    
    if (write_filename):
        with open(write_filename, 'w') as f:
            if (write_text):
                f.write(write_text+'\n')
            f.write("-"*80)
            f.write('# of questions=%d\n' %question_count)
            f.write('MR (Mean Rank)=%f\n' %mr)
            f.write('mrr v1 (without null answers)=%f\n' %mrr_v1)
            f.write('mrr v2 (with null answers)=%f\n' %mrr_v2)
            f.write('recall@1=%f\n' %recall_at_1)
            f.write('recall@2=%f\n' %recall_at_2)
            f.write('recall@3=%f\n' %recall_at_3)
            f.write('recall@4=%f\n' %recall_at_4)
            f.write('recall@5=%f\n' %recall_at_5)
            f.write('recall@10=%f\n' %recall_at_10)
            f.write('recall@20=%f\n' %recall_at_20)
            f.write('recall@50=%f\n' %recall_at_50)
            f.write('for latex: ' + str("%.3f" % round(mrr_v2, 3)) + ' & ' + str("%.3f" % round(recall_at_1, 3)) + ' & ' + str("%.3f" % round(recall_at_2, 3))
          + ' & ' + str("%.3f" % round(recall_at_3, 3)) + ' & ' + str("%.3f" % round(recall_at_4, 3)) + ' & ' + str("%.3f" % round(recall_at_5, 3))
          + ' & ' + str("%.3f" % round(recall_at_10, 3)) +'\n')
            f.write("-"*80)


def mrr_only(n, dir, json_file, is_2ndcls, is_3rdcls):
    question_count, answer_passage_ranks = get_answer_passage_ranks(dir, json_file, is_2ndcls, is_3rdcls)
    mrr_v1 = calculate_mrr_v1(answer_passage_ranks)
    mrr_v2 = calculate_mrr_v2(answer_passage_ranks)

    #print('# of questions=', question_count)
    #print('mrr v1 (mr)=', round(mrr_v1, round_dec_points))
    print('DEA' + str(n), 'mrr v2=', round(mrr_v2, round_dec_points))


def get_rank( ranked_contexts, is_2ndcls, is_3rdcls):
    rank = -1
    for context in ranked_contexts:
        if context['answer_contain']:
            if is_2ndcls:
                rank = context["2nd_cls_rank"]
            else:
                if is_3rdcls:
                    rank = context["3rd_cls_rank"]
                else:
                    rank = context["rank"]
            break
    return rank


def get_answer_passage_ranks(json_file, is_2ndcls, is_3rdcls, ctx_col):
    answer_passage_ranks = []
    f_json = open(json_file, 'r')
    print ("Evaluating...")
    f_json_lines = f_json.readlines()
    for f_json_line in tqdm(f_json_lines):
        json_obj = json.loads(f_json_line)
        ranked_contexts = json_obj[ctx_col]
        answer_passage_ranks.append(get_rank(ranked_contexts, is_2ndcls, is_3rdcls))
    f_json.close()
    print (answer_passage_ranks)
    neg_count = answer_passage_ranks.count(-1)
    print ("Negative count " +str(neg_count))
    return len(f_json_lines), answer_passage_ranks

#ranks - 3, 6, -1, 20, -1
#3+6+20+ = 29/3
#3+6+20+ = 29/5
def calculate_mr(answer_passage_ranks):
    nullAnswerCount = 0.0
    sumRanks = 0.0
    for i in range(0, len(answer_passage_ranks)):
        if answer_passage_ranks[i] != -1:
            sumRanks += answer_passage_ranks[i]
        else:
            #sumRanks += 0
            nullAnswerCount += 1
    return sumRanks / (len(answer_passage_ranks) - nullAnswerCount)
    #return sumRanks / (len(answer_passage_ranks) - 0)


def calculate_mrr_v1(answer_passage_ranks):
    nullAnswerCount = 0.0
    sumRanks = 0.0
    for i in range(0, len(answer_passage_ranks)):
        if answer_passage_ranks[i] != -1:
            #print(1/answer_passage_ranks[i])
            sumRanks += 1/answer_passage_ranks[i]
        else:
            #sumRanks += 0
            nullAnswerCount += 1
    return sumRanks / (len(answer_passage_ranks) - nullAnswerCount)
    #return sumRanks / len(answer_passage_ranks)

#ranks - 3, 6, -1, 20, -1
#0.33+0.16/4 = 0.5/4 = 0.125
#0.33+0.16/2 = 0.5/2 = 0.25
def calculate_mrr_v2_at(answer_passage_ranks, n):
    nullAnswerCount = 0.0
    sumRanks = 0.0
    for i in range(0, len(answer_passage_ranks)):
        if answer_passage_ranks[i] != -1:
            if answer_passage_ranks[i]<=n:
                #print(1/answer_passage_ranks[i])
                sumRanks += 1/answer_passage_ranks[i]
        else:
            nullAnswerCount += 1
    #return sumRanks / (len(answer_passage_ranks) - nullAnswerCount)
    return sumRanks / len(answer_passage_ranks)

#ranks - 3, 6, -1, 20, -1
#R@5 - 1/3 
#R@5 - 1/5
def calculate_recall_at(answer_passage_ranks, n):
    nullAnswerCount = 0.0
    numAnswersAT = 0.0
    for i in range(0, len(answer_passage_ranks)):
        if answer_passage_ranks[i] != -1:
            if answer_passage_ranks[i] <= n:
                numAnswersAT += 1
        else:
            nullAnswerCount += 1
    #return numAnswersAT / (len(answer_passage_ranks) - nullAnswerCount)
    return numAnswersAT / (len(answer_passage_ranks))



    
