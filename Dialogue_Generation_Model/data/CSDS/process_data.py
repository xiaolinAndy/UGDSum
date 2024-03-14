import json

final_data = {'train': [], 'valid': [], 'test': []}
kv = {'train': 'train', 'val': 'valid', 'test': 'test'}
for name in ['train', 'val', 'test']:
    file_path = 'raw/' + name + '.json'
    with open(file_path, 'r') as f:
        data = json.load(f)
    for sample in data:
        context = []
        for turn in sample['Dialogue']:
            tmp_utt = ''
            if turn['speaker'] == 'Q':
                tmp_utt += sample['QRole'] + ':'
            else:
                tmp_utt += '客服' + ':'
            tmp_utt += ''.join(turn['utterance'].split())
            context.append(' '.join(list(tmp_utt)))
        for i in range(2, len(context) + 1):
            final_data[kv[name]].append(context[:i])
with open('CSDS_dial.json', 'w') as f:
     json.dump(final_data, f, indent=4, ensure_ascii=False)

#get ppl data
final_data = {'train': [], 'valid': [], 'test': []}
final_count = {'train': [], 'valid': [], 'test': []}
kv = {'train': 'train', 'val': 'valid', 'test': 'test'}
for name in ['test']:
    file_path = 'raw/' + name + '.json'
    with open(file_path, 'r') as f:
        data = json.load(f)
    for sample in data[:10]:
        counts = []
        context = []
        for turn in sample['Dialogue']:
            tmp_utt = ''
            if turn['speaker'] == 'Q':
                tmp_utt += sample['QRole'] + ':'
            else:
                tmp_utt += '客服' + ':'
            tmp_utt += ''.join(turn['utterance'].split())
            context.append(' '.join(list(tmp_utt)))
        for i in range(0, len(context) - 1):
            count = 0
            history = context[:i]
            history_tmp = context[:i+1]
            for j in range(1, 6):
                if i + j < len(context):
                    history.append(context[i + j])
                    history_tmp.append(context[i + j])
                    if len(history) > 1:
                        count += 2
                        final_data[kv[name]].append(history.copy())
                        final_data[kv[name]].append(history_tmp.copy())
            counts.append(count)
        final_count[kv[name]].append(counts)
with open('CSDS_dial_ppl.json', 'w') as f:
     json.dump(final_data, f, indent=4, ensure_ascii=False)
with open('CSDS_dial_count.json', 'w') as f:
    json.dump(final_count, f, indent=4, ensure_ascii=False)

#get relativeness data
MAX_DIST = 15

final_data = {'train': [], 'valid': [], 'test': []}
final_count = {'train': [], 'valid': [], 'test': []}
kv = {'train': 'train', 'val': 'valid', 'test': 'test'}
for name in ['test', 'val']:
    file_path = 'raw/' + name + '.json'
    with open(file_path, 'r') as f:
        data = json.load(f)
    for sample in data:
        context = []
        for turn in sample['Dialogue']:
            tmp_utt = ''
            if turn['speaker'] == 'Q':
                tmp_utt += sample['QRole'] + ':'
            else:
                tmp_utt += '客服' + ':'
            tmp_utt += ''.join(turn['utterance'].split())
            context.append(' '.join(list(tmp_utt)))
        for i in range(len(context)):
            history = context[:i]
            history_tmp = context[:i+1]
            for j in range(i + 1, min(len(context), i + MAX_DIST)):
                history.append(context[j])
                history_tmp.append(context[j])
                final_data[kv[name]].append(history.copy())
                final_data[kv[name]].append(history_tmp.copy())
        final_count[kv[name]].append(len(context))
with open('CSDS_dial_ppl.json', 'w') as f:
    json.dump(final_data, f, indent=4, ensure_ascii=False)
with open('CSDS_dial_count.json', 'w') as f:
    json.dump(final_count, f, indent=4, ensure_ascii=False)

#get idf data
final_data = {'train': []}
for name in ['train']:
    file_path = 'raw/' + name + '.json'
    with open(file_path, 'r') as f:
        data = json.load(f)
    for sample in data:
        context = []
        for turn in sample['Dialogue']:
            tmp_utt = ''
            if turn['speaker'] == 'Q':
                tmp_utt += sample['QRole'] + ':'
            else:
                tmp_utt += '客服' + ':'
            tmp_utt += ''.join(turn['utterance'].split())
            context.append(' '.join(list(tmp_utt)))
        final_data[name].append(context)

with open('CSDS_idf_dial.json', 'w') as f:
     json.dump(final_data, f, indent=4, ensure_ascii=False)

#get lm data

final_data = {'train': [], 'valid': [], 'test': []}
kv = {'train': 'train', 'val': 'valid', 'test': 'test'}
for name in ['test', 'val']:
    file_path = 'raw/' + name + '.json'
    with open(file_path, 'r') as f:
        data = json.load(f)
    for sample in data:
        context = []
        for turn in sample['Dialogue']:
            tmp_utt = ''
            if turn['speaker'] == 'Q':
                tmp_utt += sample['QRole'] + ':'
            else:
                tmp_utt += '客服' + ':'
            tmp_utt += ''.join(turn['utterance'].split())
            context.append(' '.join(list(tmp_utt)))
        for i in range(len(context)):
            final_data[kv[name]].append([context[i]])
with open('CSDS_dial_lm.json', 'w') as f:
    json.dump(final_data, f, indent=4, ensure_ascii=False)
