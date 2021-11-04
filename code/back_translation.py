from deep_translator import GoogleTranslator

from tqdm.auto import tqdm
import pandas as pd
from datasets import load_from_disk

import os

import argparse


def google_translate(sent):
    ko_to_en = GoogleTranslator(source='auto', target='en')
    en_to_ko = GoogleTranslator(source='auto', target='ko')
    try:
        en_sent = ko_to_en.translate(sent)
        ko_sent = en_to_ko.translate(en_sent)
    except:
        return sent
    return ko_sent


def translate_bulk(sent_list):
    new_sent = []
    for idx in tqdm(range(len(sent_list)), desc="Tons of Translations Ongoing..."):
        trans_sent = google_translate(sent_list[idx])
        new_sent.append(trans_sent)
    return new_sent


def translate_sentence(sentence):
    trans_sent = google_translate(sentence)
    return trans_sent


train_dataset = load_from_disk("../data/train_dataset/")['train']
print("Train Dataset Loaded!")

context = list(train_dataset['context'])
question = list(train_dataset['question'])
answers = list(train_dataset['answers'])
index_level = list(train_dataset['__index_level_0__'])
document_id = list(train_dataset['document_id'])
ids = list(train_dataset['id'])
title = list(train_dataset['title'])

def match_answer(sent, answer):
    for idx in range(len(sent)):
        if sent[idx:idx+len(answer)] == answer:
            return idx


def proper_translate(corpus, answer_dict):

    sentences = corpus.split('.')
    sentences = [(sent+'.', len(sent)+1) for sent in sentences]
    reference = [False for _ in range(len(sentences))]
    new_corpus = []
    answer, start_idx = answer_dict['text'][0], answer_dict['answer_start'][0]
    idx_count, new_idx = 0, 0
    
    for idx, sent in enumerate(sentences):
        if start_idx in range(idx_count, idx_count+sent[1]) and answer in sent[0]:
            new_idx += match_answer(sent[0], answer)
            reference[idx] = True
            break
        idx_count += sent[1]

    idx_accumulate = True
    for idx in range(len(reference)):
        if reference[idx] is True:
            new_corpus.append(sentences[idx][0]+' ')
            idx_accumulate = False
        else:
            trans_sent = translate_sentence(sentences[idx][0])
            new_corpus.append(trans_sent+' ')
            if idx_accumulate:
                new_idx += len(trans_sent)+1
        
    return ''.join(new_corpus)[:-3], new_idx

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default="../data/back_translation/")
parser.add_argument('--start_idx', type=int, default=0)
parser.add_argument('--end_idx', type=int, default=5000)
args = parser.parse_args()

save_dir = args.save_dir
start_idx = args.start_idx
end_idx = args.end_idx

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

if end_idx > train_dataset.num_rows-1:
    end_idx = train_dataset.num_rows-1

new_answers = []
new_context = []

for idx in tqdm(range(start_idx, end_idx), desc="Back Translation On-going..."):
    new_txt, new_id = proper_translate(context[idx], answers[idx])
    new_answer = {"answer_start":[new_id], "text":answers[idx]['text']}
    new_answers.append(new_answer)
    new_context.append(new_txt)

# features: ['__index_level_0__', 'answers', 'context', 'document_id', 'id', 'question', 'title']
df = pd.DataFrame({'__index_level_0__':index_level[start_idx:end_idx],
                    'answers':new_answers,
                    'context':new_context,
                    'document_id':ids[start_idx:end_idx],
                    'question':question[start_idx:end_idx],
                    'title':title[start_idx:end_idx]})
df.to_csv(save_dir + f"trans{start_idx}_{end_idx}.csv", index=False)
print(f"trans{start_idx}_{end_idx}.csv saved to {save_dir} !!!")