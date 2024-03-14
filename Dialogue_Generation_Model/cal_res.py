import json
import numpy as np

with open('data/CSDS/CSDS_dial_ppl.json', 'r') as f:
    dial = json.load(f)['test']
with open('data/CSDS/CSDS_dial_count.json', 'r') as f:
    count = json.load(f)['test']
with open('CSDS_result_72.txt', 'r') as f:
    ppl = json.load(f)
with open('data/CSDS/raw/test.json', 'r') as f:
    raw_data = json.load(f)

index = 0
for k, s in enumerate(count):
    scores = []
    for i in range(s):
        score = []
        for j in range(i + 1, s):
            score_without = ppl[index]
            index += 1
            score_with = ppl[index]
            index += 1
            score.append(score_without - score_with)
        scores.append(score)
    cents = []
    for i in range(s):
        max_score = -1e9
        max_index = -1
        tmp_scores = []
        for j in range(s):
            if i < j:
                tmp_scores.append(scores[i][j - i - 1])
                if max_score < scores[i][j - i - 1]:
                    max_score = scores[i][j - i - 1]
                    max_index = j
            elif i > j:
                tmp_scores.append(scores[j][i-j-1])
                if max_score < scores[j][i-j-1]:
                    max_score = scores[j][i-j-1]
                    max_index = j
        # print(raw_data[k+2]['Dialogue'][i]['utterance'])
        # print(raw_data[k+2]['Dialogue'][max_index]['utterance'])
        cents.append(-np.mean(np.array(tmp_scores)))
    cents = np.array(cents)
    chosen = np.argpartition(cents, 5)
    for i in chosen[:5]:
        print(raw_data[k]['Dialogue'][i]['utterance'])