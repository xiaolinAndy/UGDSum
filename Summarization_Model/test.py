import json
import os
import nltk
import numpy as np

from tqdm import tqdm
from summa import summarizer
from evaluate import get_rouge
from metric import compute_rouge_n

#dataset = 'SAMSum'
dataset = 'CSDS'
mode = 'overall'
# dataset = 'AMI'

def get_longest(dial, max_length):
    if dataset == 'SAMSum':
        lengths = [-len(nltk.word_tokenize(s)) for s in dial]
    else:
        lengths = [-len(''.join(s.split())) for s in dial]
    # if len(lengths) > 3:
    #     indexes = np.argpartition(lengths, 3)[:3]
    # else:
    #     indexes = [i for i in range(len(lengths))]
    # indexes.sort()
    # return [dial[s] for s in indexes]
    indexes = np.argsort(lengths)
    cand = []
    total_len = 0
    for i in indexes:
        total_len -= lengths[i]
        cand.append(i)
        if total_len >= max_length and len(cand) > 0:
            break
    cand.sort()
    sum = [dial[s] for s in cand]
    return sum

def get_lead(dial, max_length, mode):
    if dataset == 'SAMSum':
        lengths = [-len(nltk.word_tokenize(s)) for s in dial]
    else:
        lengths = [-len(''.join(s.split())) for s in dial]
    indexes = list(range(len(dial)))
    cand = []
    total_len = 0
    for i in indexes:
        if dial[i].split()[0] == '客服' and mode == 'user' or dial[i].split()[0] != '客服' and mode == 'agent':
            continue
        total_len -= lengths[i]
        cand.append(i)
        if total_len >= max_length and len(cand) > 0:
            break
    cand.sort()
    sum = [dial[s] for s in cand]
    return sum

def get_oracle(dial, sum, max_length):
    if dataset == 'SAMSum' or dataset == 'AMI':
        lengths = [-len(nltk.word_tokenize(s)) for s in dial]
        rouge_scores = [-compute_rouge_n(nltk.word_tokenize(s), nltk.word_tokenize(sum)) for s in dial]
    else:
        lengths = [-len(''.join(s.split())) for s in dial]
        rouge_scores = [-compute_rouge_n(''.join(s.split()), ''.join(sum.split())) for s in dial]
    indexes = np.argsort(rouge_scores)
    cand = []
    total_len = 0
    for i in indexes:
        # if dial[i].split()[0] == '客服' and mode == 'user' or dial[i].split()[0] != '客服' and mode == 'agent':
        #     continue
        total_len -= lengths[i]
        cand.append(i)
        if total_len >= max_length and len(cand) > 0:
            break
    cand.sort()
    sum = [dial[s] for s in cand]
    #return sum, [-s for s in rouge_scores], [int(s) for s in cand]
    return sum

def get_oracle_recall(dial, sum, max_length):
    if dataset == 'SAMSum':
        lengths = [-len(nltk.word_tokenize(s)) for s in dial]
        rouge_scores = [-compute_rouge_n(nltk.word_tokenize(s), nltk.word_tokenize(sum), mode='r') for s in dial]
    else:
        lengths = [-len(''.join(s.split())) for s in dial]
        rouge_scores = [-compute_rouge_n(''.join(s.split()), ''.join(sum.split()), mode='r') for s in dial]
    indexes = np.argsort(rouge_scores)
    cand = []
    total_len = 0
    for i in indexes:
        total_len -= lengths[i]
        cand.append(i)
        if total_len >= max_length and len(cand) > 0:
            break
    cand.sort()
    sum = [dial[s] for s in cand]
    #return sum, [-s for s in rouge_scores], [int(s) for s in cand]
    return sum

# overall
if mode == 'overall':
    if dataset == 'CSDS':
        data_path = 'data/CSDS/sum/test.json'
        #nll_path = '../../dialogue_generation/CDial-GPT-master/data/CSDS/nll_test_res.json'
        #nll_path = '../../dialogue_generation/CDial-GPT-master/data/CSDS/nll_new_test_res.json'
        #nll_path = '../../dialogue_generation/CDial-GPT-master/data/CSDS/nll_idf_test_res.json'
        #nll_reverse_path = '../../dialogue_generation/CDial-GPT-master/data/CSDS/nll_reverse_test_res.json'
        #nll_path = '../../dialogue_generation/CDial-GPT-master/data/CSDS/nll_org_test_res.json'
        #nll_path = '../../dialogue_generation/dialogue-hred-vhred/data/csds_char/nll_mask_idf_dial_test_res_1.json'
        #nll_path = '../../dialogue_generation/dialogue-hred-vhred/data/csds/nll/nll_mask_idf_dial_test_res.json'
        #nll_path = '../../dialogue_generation/dialogue-hred-vhred/data/csds_first/nll_idf_dial_test_res.json'
        # nll_path = '../../dialogue_generation/dialogue-hred-vhred/data/csds_first/hred_uni_nll_idf_dial_test_res.json'
        #weight_path = '../../dialogue_generation/dialogue-hred-vhred/data/csds/nll/lm_weight_test.json'
        #weight_path = '../../dialogue_generation/CDial-GPT-master/data/CSDS/lm_idf_weight_filter_test.json'
        #weight_path = '../../dialogue_generation/CDial-GPT-master/data/CSDS/lm_weight_filter_test.json'
        #weight_path = '../../dialogue_generation/dialogue-hred-vhred/data/csds_first/lm_weight_filter_test.json'
        # weight_path = '../../dialogue_generation/dialogue-hred-vhred/data/csds_first/hred_uni_lm_weight_filter_test.json'
        # pair_weight_path = 'test_weights_pair.json'
        nll_path = '../Dialogue_Generation_Model/result/CSDS_ppl_test.json'
        weight_path = '../Dialogue_Generation_Model/result/CSDS_lm_test.json'

        with open(data_path, 'r') as f:
            data = json.load(f)

        with open(nll_path, 'r') as f:
            nll = json.load(f)

        # with open(nll_reverse_path, 'r') as f:
        #     nll_reverse = json.load(f)

        with open(weight_path, 'r') as f:
            weight = json.load(f)

        with open(pair_weight_path, 'r') as f:
            pair_weight = json.load(f)

        dials = [s['Dialogue'] for s in data]
        refs = [s['FinalSumm'] for s in data]
        all_scores = []
        all_indexes = []

        for l in range(90, 95, 15):
            print(l, '--------------------')
            preds = []
            for i, dial in tqdm(enumerate(dials)):
                dial = '|||'.join(dial)
                sums, scores, indexes = summarizer.summarize(dial, language='chinese', length=l, nll=nll[i], weight=weight[i], num=5, lamb=0.7, nll_re=None)
                #sums, scores, indexes = summarizer.summarize(dial, language='chinese', length=l, nll=None, num=5, weight=None, pair_weight=None)
                #sums = get_longest(dial, max_length=l)
                #sums, scores, indexes = get_oracle(dial, refs[i], max_length=l)
                #sums= get_lead(dial, max_length=l)
                sums = ''.join([''.join(s.split()) for s in sums])
                preds.append(sums)
                # all_scores.append(scores)
                # #print(indexes)
                all_indexes.append(indexes)

            #with open('result/CSDS_textrank_filter_test_preds.txt', 'w') as f:
            with open('result/CSDS_pred.txt', 'w') as f:
                for pred in preds:
                    f.write(pred + '\n')

            #get_rouge('result/CSDS_textrank_filter_test_preds.txt', 'data/CSDS/final_refs_test.txt')
            get_rouge('result/CSDS_pred.txt', 'data/CSDS/final_refs_test.txt')
            #get_rouge('result/CSDS_debug_preds.txt', 'data/CSDS/final_refs_val.txt')

            # with open('result/CSDS_oracle_recall_scores.json', 'w') as f:
            #     json.dump(all_scores, f, indent=4)
            #
            # with open('result/CSDS_UGD_indexes.json', 'w') as f:
            #     json.dump(all_indexes, f, indent=4)
    elif dataset == 'SAMSum':
        data_path = 'data/SAMSum/sum/test.json'
        #nll_path = '../../dialogue_generation/CDial-GPT-master/data/SAMSum/nll_test_res.json'
        #nll_path = '../../dialogue_generation/CDial-GPT-master/data/SAMSum/nll_new_test_res.json'
        #nll_path = '../../dialogue_generation/CDial-GPT-master/data/SAMSum/nll_idf_test_res.json'
        #nll_path = '../../dialogue_generation/CDial-GPT-master/data/SAMSum/nll_dial_test_res.json'
        # nll_path = '../../dialogue_generation/dialogue-hred-vhred/data/ppl_samsum/nll_dial_test_res.json'
        #nll_path = '../../dialogue_generation/dialogue-hred-vhred/data/samsum_first/nll_mask_idf_dial_test_res.json'
        nll_path = '../../dialogue_generation/dialogue-hred-vhred/data/samsum_first/hred_uni_nll_mask_idf_dial_test_res.json'
        #weight_path = '../../dialogue_generation/CDial-GPT-master/data/SAMSum/lm_weight_test.json'
        #weight_path = '../../dialogue_generation/dialogue-hred-vhred/data/samsum_first/lm_weight_filter_test.json'
        #weight_path = '../../dialogue_generation/dialogue-hred-vhred/data/lm_samsum/lm_weight_filter_test.json'
        weight_path = '../../dialogue_generation/dialogue-hred-vhred/data/samsum_first/hred_uni_lm_weight_filter_test.json'


        with open(data_path, 'r') as f:
            data = json.load(f)

        with open(nll_path, 'r') as f:
            nll = json.load(f)

        with open(weight_path, 'r') as f:
            weight = json.load(f)

        dials = [s['Dialogue'] for s in data]
        refs = [s['Sum'] for s in data]
        all_scores = []
        all_indexes = []

        for l in range(50, 55, 10):
            print(l, '--------------------')
        # l = 55
            preds = []
            for i, dial in tqdm(enumerate(dials)):
                dial = '|||'.join(dial)
                #sums, scores, indexes = summarizer.summarize(dial, language='english', length=l)
                sums, scores, indexes = summarizer.summarize(dial, language='english', length=l, nll=nll[i], weight=weight[i], lamb=0.9)
                #sums = get_longest(dial, max_length=l)
                #sums, scores, indexes = get_oracle(dial, refs[i], max_length=l)
                # sums = get_lead(dial, max_length=l)
                sums = [' '.join(nltk.word_tokenize(s)) for s in sums]
                sums = ' '.join(sums)
                preds.append(sums)
                # all_scores.append(scores)
                indexes.sort()
                all_indexes.append(indexes)

            #with open('result/SAMSum_textrank_filter_test_preds.txt', 'w') as f:
            with open('result/SAMSum_hred_uni.txt', 'w') as f:
                for pred in preds:
                    f.write(pred + '\n')

            # os.system('files2rouge data/SAMSum/final_refs_test_split.txt result/SAMSum_debug_preds.txt -s rouge.txt')
            os.system('files2rouge data/SAMSum/final_refs_test_split.txt result/SAMSum_hred_uni.txt -s rouge.txt')
            #os.system('files2rouge data/SAMSum/final_refs_val_split.txt result/SAMSum_debug_preds.txt -s rouge.txt')

            # with open('result/SAMSum_oracle_scores.json', 'w') as f:
            #     json.dump(all_scores, f, indent=4)
            #
            # with open('result/SAMSum_UGD_indexes.json', 'w') as f:
            #     json.dump(all_indexes, f, indent=4)

    elif dataset == 'AMI':
        data_path = 'data/AMI/test.json'
        nll_path = '../../dialogue_generation/CDial-GPT-master/data/AMI/nll_dial_test_res_3.json'
        weight_path = '../../dialogue_generation/CDial-GPT-master/data/AMI/lm_weight_test_3.json'


        with open(data_path, 'r') as f:
            data = json.load(f)

        with open(nll_path, 'r') as f:
            nll = json.load(f)

        with open(weight_path, 'r') as f:
            weight = json.load(f)

        dials = [s['Dialogue'] for s in data]
        refs = [s['Sum'] for s in data]
        all_scores = []
        all_indexes = []

        # for l in range(250, 260, 10):
        #     print(l, '--------------------')
        l = 250
        for la in [0.3]:
            print(la, '--------------------')
            preds = []
            for i, dial in tqdm(enumerate(dials)):
                dial = '|||'.join(dial)
                #sums, scores, indexes = summarizer.summarize(dial, language='english', length=l, lamb=1)
                sums, scores, indexes = summarizer.summarize(dial, language='english', length=l, nll=nll[i], weight=weight[i], lamb=la/10)
                #sums = get_longest(dial, max_length=l)
                #sums = get_oracle(dial, refs[i], max_length=l)
                # sums = get_lead(dial, max_length=l)
                sums = [' '.join(nltk.word_tokenize(s)) for s in sums]
                sums = ' '.join(sums)
                preds.append(sums)
                # all_scores.append(scores)
                # indexes.sort()
                # all_indexes.append(indexes)

            with open('result/AMI_preds.txt', 'w') as f:
                for pred in preds:
                    f.write(pred + '\n')

            os.system('files2rouge data/AMI/test_ref_split.txt result/AMI_preds.txt -s rouge.txt')

elif mode == 'user':
    data_path = 'data/CSDS/sum/test.json'
    nll_path = '../../dialogue_generation/CDial-GPT-master/data/CSDS/nll_idf_test_res.json'
    weight_path = '../../dialogue_generation/CDial-GPT-master/data/CSDS/lm_idf_weight_filter_test.json'
    #data_path = 'data/CSDS/sum/val.json'
    with open(data_path, 'r') as f:
        data = json.load(f)
    with open(nll_path, 'r') as f:
        nll = json.load(f)
    with open(weight_path, 'r') as f:
        weight = json.load(f)
    dials = [s['Dialogue'] for s in data]
    refs = [s['UserSumm'] for s in data]

    for l in range(30, 35, 5):#30
        print(l, '--------------------')
        preds = []
        for i, dial in tqdm(enumerate(dials)):
            #dial = '|||'.join(dial)
            # sums, scores, indexes = summarizer.summarize(dial, language='chinese', length=l, nll=None, num=5, weight=None,
            #                                              pair_weight=None, mode='user')
            # sums, scores, indexes = summarizer.summarize(dial, language='chinese', length=40, nll=nll[i], weight=weight[i],
            #                                              num=5, lamb=0.7, nll_re=None, mode='user')
            sums = get_oracle(dial, refs[i], max_length=l, mode='user')
            sums = ''.join([''.join(s.split()) for s in sums])
            preds.append(sums)
        with open('result/CSDS_debug_user_preds.txt', 'w') as f:
            for pred in preds:
                f.write(pred + '\n')
        get_rouge('result/CSDS_debug_user_preds.txt', 'data/CSDS/user_refs_test_split.txt')
        #get_rouge('result/CSDS_debug_user_preds.txt', 'data/CSDS/user_refs_val_split.txt')

    refs = [s['AgentSumm'] for s in data]

    for l in range(45, 50, 5):#45
        print(l, '--------------------')
        preds = []
        for i, dial in tqdm(enumerate(dials)):
            #dial = '|||'.join(dial)
            # sums, scores, indexes = summarizer.summarize(dial, language='chinese', length=l, nll=None, num=5, weight=None,
            #                                              pair_weight=None, mode='agent')
            # sums, scores, indexes = summarizer.summarize(dial, language='chinese', length=40, nll=nll[i], weight=weight[i],
            #                                              num=5, lamb=0.7, nll_re=None, mode='agent')
            sums = get_oracle(dial, refs[i], max_length=l, mode='agent')
            sums = ''.join([''.join(s.split()) for s in sums])
            preds.append(sums)
        with open('result/CSDS_debug_agent_preds.txt', 'w') as f:
            for pred in preds:
                f.write(pred + '\n')
        get_rouge('result/CSDS_debug_agent_preds.txt', 'data/CSDS/agent_refs_test_split.txt')
        #get_rouge('result/CSDS_debug_agent_preds.txt', 'data/CSDS/agent_refs_val_split.txt')




