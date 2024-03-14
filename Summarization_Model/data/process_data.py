import json
import re
import nltk


def get_hit(dial, sum):
    hit = 0
    for s in sum:
        if s not in dial:
            hit += 1
    return hit, len(sum)


for name in ['train', 'val', 'test']:
    sum_tokens = 0
    dial_utts = 0
    dial_tokens = 0
    new_one, new_two = 0, 0
    all_one, all_two = 0, 0
    file_path = 'raw/' + name + '.json'
    with open(file_path, 'r') as f:
        data = json.load(f)
    new_data = []
    for sample in data:
        one_grams, two_grams = [], []
        context = []
        for turn in sample['Dialogue']:
            tmp_utt = ''
            if turn['speaker'] == 'Q':
                tmp_utt += sample['QRole'] + ' : '
            else:
                tmp_utt += '客服' + ' : '
            tmp_utt += ''.join(turn['utterance'].split())
            if tmp_utt[-1] not in ['。', '？', '！', '.', '?', '!']:
                tmp_utt += ' 。'
            one_grams += list(tmp_utt)
            two_grams += [''.join(tmp_utt[i:i+2]) for i in range(len(tmp_utt))]
            dial_tokens += len(tmp_utt)
            context.append(tmp_utt)
        dial_utts += len(context)
        final_sum = ''.join(sample['FinalSumm'])
        user_sum = ''.join(sample['UserSumm'])
        agent_sum = ''.join(sample['AgentSumm'])
        new_data.append({'Dialogue': context,
                         'Sum': final_sum,
                         'usersumm': user_sum,
                         'agentsumm': agent_sum})
        one_grams = list(set(one_grams))
        two_grams = list(set(two_grams))
        one_hit, one_all = get_hit(one_grams, list(final_sum))
        two_hit, two_all = get_hit(two_grams, [final_sum[i:i+2]for i in range(len(final_sum) - 1)])
        new_one += one_hit
        new_two += two_hit
        all_one += one_all
        all_two += two_all
        sum_tokens += len(''.join(sample['FinalSumm']))
    print(name)
    print('data size: ', len(data))
    print('sum length: ', sum_tokens / len(data))
    print('dial utt: ', dial_utts / len(data))
    print('dial length: ', dial_tokens / len(data))
    print('compress ratio: ', sum_tokens / dial_tokens)
    print('new one: ', new_one / all_one)
    print('new two: ', new_two / all_two)
    with open('sum/' + name + '.json', 'w') as f:
         json.dump(new_data, f, indent=4, ensure_ascii=False)
    if name == 'test':
        # with open('final_refs_test_split.txt', 'w') as f:
        #     for sum in [s['Sum'] for s in new_data]:
        #         f.write(' '.join(nltk.word_tokenize(sum)) + '\n')
        with open('user_refs_test_split.txt', 'w') as f:
            for sum in [s['usersumm'] for s in new_data]:
                f.write(' '.join(nltk.word_tokenize(sum)) + '\n')
        with open('agent_refs_test_split.txt', 'w') as f:
            for sum in [s['agentsumm'] for s in new_data]:
                f.write(' '.join(nltk.word_tokenize(sum)) + '\n')
    if name == 'val':
    #     with open('final_refs_val.txt', 'w') as f:
    #         for sum in [s['Sum'] for s in new_data]:
    #             f.write(sum + '\n')
        with open('user_refs_val_split.txt', 'w') as f:
            for sum in [s['usersumm'] for s in new_data]:
                f.write(' '.join(nltk.word_tokenize(sum)) + '\n')
        with open('agent_refs_val_split.txt', 'w') as f:
            for sum in [s['agentsumm'] for s in new_data]:
                f.write(' '.join(nltk.word_tokenize(sum)) + '\n')

