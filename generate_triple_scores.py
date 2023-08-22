from tqdm import tqdm

def Jaccard_Similarity(doc1, doc2, threshold=0.0): 
    #ref: https://studymachinelearning.com/jaccard-similarity-text-similarity-metric-in-nlp/
    # List the unique words in a document
    words_doc1 = set(doc1.lower().split()) 
    words_doc2 = set(doc2.lower().split())
    
    # Find the intersection of words list of doc1 & doc2
    intersection = words_doc1.intersection(words_doc2)

    # Find the union of words list of doc1 & doc2
    union = words_doc1.union(words_doc2)
    
    ##IF Both Strings are empty return 0
    if(len(union)==0):
        return 0.0
    
    # Calculate Jaccard similarity score 
    # using length of intersection set divided by length of union set


    jaccard_score = float(len(intersection)) / len(union)
    if(threshold>0.0):
        if (jaccard_score>threshold):
            return 1.0
        else:
            return 0.0 
    return jaccard_score

def calculate_overlap_scores(s1):
    subject_overlap_scores = []
    predicate_overlap_scores = []
    object_overlap_scores = []
    overall_overlap_scores = []


    subject_overlap_scores_b = []
    predicate_overlap_scores_b = []
    object_overlap_scores_b = []
    overall_overlap_scores_b = []

    subject_coverage_scores = []
    predicate_coverage_scores = []
    object_coverage_scores = []
    overall_coverage_scores = []

    for i in tqdm(range(len(s1))):
        jaccard_subject_score = 0
        jaccard_predicate_score = 0
        jaccard_object_score = 0
        jaccard_overall_score = 0

        jaccard_subject_score_b = 0
        jaccard_predicate_score_b = 0
        jaccard_object_score_b = 0
        jaccard_overall_score_b = 0
        
        jaccard_subject_coverage_score = 0
        jaccard_predicate_coverage_score = 0
        jaccard_object_coverage_score = 0
        jaccard_overall_coverage_score = 0

        jaccard_subject_scores_list = []
        jaccard_predicate_scores_list = []
        jaccard_object_scores_list = []
        jaccard_overall_scores_list = []

        jaccard_subject_scores_list_b = []
        jaccard_predicate_scores_list_b = []
        jaccard_object_scores_list_b = []
        jaccard_overall_scores_list_b = []

        jaccard_subject_coverage_scores_list = []
        jaccard_predicate_coverage_scores_list = []
        jaccard_object_coverage_scores_list = []
        jaccard_overall_coverage_scores_list = []

        ctx_triples=s1['triples'].iloc[i]
        question_triples = s1['query_triples'].iloc[i]

        for question_triple in question_triples:
            question_subject = question_triple[0]
            question_predicate = question_triple[1]
            question_object = question_triple[2]

            question_subject_coverage = 0
            question_predicate_coverage = 0
            question_object_coverage = 0
            question_overall_coverage = 0

            for ctx_triple in ctx_triples:
                ctx_subject = ctx_triple[0]
                ctx_predicate = ctx_triple[1]
                ctx_object = ctx_triple[2]

                temp_subject_score = Jaccard_Similarity(ctx_subject, question_subject)
                temp_predicate_score = Jaccard_Similarity(ctx_predicate, question_predicate)
                temp_object_score = Jaccard_Similarity(ctx_object, question_object)

                jaccard_subject_scores_list.append(temp_subject_score)
                jaccard_predicate_scores_list.append(temp_predicate_score)
                jaccard_object_scores_list.append(temp_object_score)
                temp_average_score = (temp_subject_score + temp_predicate_score + temp_object_score) / 3
                jaccard_overall_scores_list.append(temp_average_score)

                ###Threshold based method
                threshold = 0.8
                temp_subject_b_score = Jaccard_Similarity(ctx_subject, question_subject, threshold=threshold)
                temp_predicate_b_score = Jaccard_Similarity(ctx_predicate, question_predicate, threshold=threshold)
                temp_object_b_score = Jaccard_Similarity(ctx_object, question_object, threshold=threshold)

                jaccard_subject_scores_list_b.append(temp_subject_b_score)
                jaccard_predicate_scores_list_b.append(temp_predicate_b_score)
                jaccard_object_scores_list_b.append(temp_object_b_score)

                if(temp_subject_b_score==1 and temp_predicate_b_score==1 and temp_object_b_score==1):
                    jaccard_overall_scores_list_b.append(1)
                else:
                    jaccard_overall_scores_list_b.append(0)

                ###Till Here

                ###Coverage based method
                if(temp_subject_b_score==1):
                    question_subject_coverage = 1
                if(temp_predicate_b_score==1):
                    question_predicate_coverage = 1
                if(temp_object_b_score==1):
                    question_object_coverage = 1


            jaccard_subject_coverage_scores_list.append(question_subject_coverage)
            jaccard_predicate_coverage_scores_list.append(question_predicate_coverage)
            jaccard_object_coverage_scores_list.append(question_object_coverage)

            if(question_subject_coverage==1 and question_predicate_coverage==1 and question_object_coverage==1):
                jaccard_overall_coverage_scores_list.append(1)
            else:
                jaccard_overall_coverage_scores_list.append(0)

            ###Till Here

        if(len(jaccard_subject_scores_list)>0):
            jaccard_subject_score = sum(jaccard_subject_scores_list) / len(jaccard_subject_scores_list)
        if(len(jaccard_predicate_scores_list)>0):
            jaccard_predicate_score = sum(jaccard_predicate_scores_list) / len(jaccard_predicate_scores_list)
        if(len(jaccard_object_scores_list)>0):
            jaccard_object_score = sum(jaccard_object_scores_list) / len(jaccard_object_scores_list)
        if(len(jaccard_overall_scores_list)>0):
            jaccard_overall_score = sum(jaccard_overall_scores_list) / len(jaccard_overall_scores_list)

      ###Threshold based method

        if(len(jaccard_subject_scores_list_b)>0):
            jaccard_subject_score_b = sum(jaccard_subject_scores_list_b) / len(jaccard_subject_scores_list_b)
        if(len(jaccard_predicate_scores_list_b)>0):
            jaccard_predicate_score_b = sum(jaccard_predicate_scores_list_b) / len(jaccard_predicate_scores_list_b)
        if(len(jaccard_object_scores_list_b)>0):
            jaccard_object_score_b = sum(jaccard_object_scores_list_b) / len(jaccard_object_scores_list_b)
        if(len(jaccard_overall_scores_list_b)>0):
            jaccard_overall_score_b = sum(jaccard_overall_scores_list_b) / len(jaccard_overall_scores_list_b)

      ##Coverage based method

        if(len(jaccard_subject_coverage_scores_list)>0):
            jaccard_subject_coverage_score = sum(jaccard_subject_coverage_scores_list) / len(jaccard_subject_coverage_scores_list)
        if(len(jaccard_predicate_coverage_scores_list)>0):
            jaccard_predicate_coverage_score = sum(jaccard_predicate_coverage_scores_list) / len(jaccard_predicate_coverage_scores_list)
        if(len(jaccard_object_coverage_scores_list)>0):
            jaccard_object_coverage_score = sum(jaccard_object_coverage_scores_list) / len(jaccard_object_coverage_scores_list)
        if(len(jaccard_overall_coverage_scores_list)>0):
            jaccard_overall_coverage_score = sum(jaccard_overall_coverage_scores_list) / len(jaccard_overall_coverage_scores_list)


        subject_overlap_scores.append(jaccard_subject_score)
        predicate_overlap_scores.append(jaccard_predicate_score)
        object_overlap_scores.append(jaccard_object_score)
        overall_overlap_scores.append(jaccard_overall_score)

        subject_overlap_scores_b.append(jaccard_subject_score_b)
        predicate_overlap_scores_b.append(jaccard_predicate_score_b)
        object_overlap_scores_b.append(jaccard_object_score_b)
        overall_overlap_scores_b.append(jaccard_overall_score_b)

        ##Coverage based method
        subject_coverage_scores.append(jaccard_subject_coverage_score)
        predicate_coverage_scores.append(jaccard_predicate_coverage_score)
        object_coverage_scores.append(jaccard_object_coverage_score)
        overall_coverage_scores.append(jaccard_overall_coverage_score)

    #answer_contains = s1['answer_contain'].to_list()
    #s1 = pd.DataFrame()    
    s1['subject_overlap_score']=subject_overlap_scores
    s1['predicate_overlap_score']=predicate_overlap_scores
    s1['object_overlap_score']=object_overlap_scores
    s1['overall_overlap_score']=overall_overlap_scores

    s1['subject_overlap_score_b']=subject_overlap_scores_b
    s1['predicate_overlap_score_b']=predicate_overlap_scores_b
    s1['object_overlap_score_b']=object_overlap_scores_b
    s1['overall_overlap_score_b']=overall_overlap_scores_b

    s1['subject_coverage_score']=subject_coverage_scores
    s1['predicate_coverage_score']=predicate_coverage_scores
    s1['object_coverage_score']=object_coverage_scores
    s1['overall_coverage_score']=overall_coverage_scores
    #s1['answer_contain'] = answer_contains

    s1.head(10)
    return s1
    