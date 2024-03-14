import json

# MAX_DIST = 15
with open('CSDS_dial_count.json', 'r') as f:
    counts = json.load(f)['test']

with open('../../result/CSDS_ppl_test.json', 'r') as f:
    nlls = json.load(f)

res = []
index = 0
for i in range(len(counts)):
    dial_res = []
    for a in range(counts[i]):
        tmp = []
        for b in range(counts[i]):
            tmp.append(0)
        dial_res.append(tmp)

    for j in range(counts[i]):
        for k in range(j + 1, min(j + MAX_DIST, counts[i])):
            dial_res[j][k] = nlls[index] - nlls[index + 1]
            dial_res[k][j] = nlls[index] - nlls[index + 1]
            index += 2
    res.append(dial_res)

assert index == len(nlls)

# lm

with open('raw/test.json', 'r') as f:
    data = json.load(f)


with open('../../result/CSDS_lm_test.json', 'r') as f:
    lms = json.load(f)

index = 0
dial_lms = []
for i in range(len(counts)):
    tmp_lms = []
    for j in range(counts[i]):
        if len(data[i]['Dialogue'][j]['utterance'].split()) < 3:
            tmp_lms.append(-10000)
        else:
            tmp_lms.append(lms[index])
        index += 1
    dial_lms.append(tmp_lms)

with open('lm_weight_filter_test.json', 'w') as f:
    json.dump(dial_lms, f, indent=4, ensure_ascii=False)


