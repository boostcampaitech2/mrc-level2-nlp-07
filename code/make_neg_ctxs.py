'''
BM25 를 활용해 query 와 연관성은 높지만, answer 를 포함하지 않은 문장들을 뽑아 
hard_negative contexts를 생성해주는 파일입니다.
'''

from rank_bm25 import BM25Okapi, BM25Plus
from elasticsearch import Elasticsearch

from transformers import AutoTokenizer
import json
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

from preprocess import preprocess_retrieval

tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

## wikipedia_documents.json
with open("../data/wikipedia_documents.json", "r", encoding="utf-8") as f:
    wiki = json.load(f)
contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))

## train dataset
data = pd.read_csv("../data/negative_samples.csv")
train_context = data["context"].tolist()
train_query = data["question"].tolist()
train_answers = data["answers"].tolist()
train_answers = [eval(i)["text"][0] for i in train_answers]

## prepare BM25
def prepare_bm25(contexts=contexts, model_name="monologg/koelectra-base-v3-discriminator"):
        
    contexts = list(set([example for example in contexts]))
    print("Tokenizing & Creating BM25 started!")
    preprocessed_contexts = [preprocess_retrieval(corpus) for corpus in tqdm(contexts)]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_wiki = [tokenizer.tokenize(corpus) for corpus in tqdm(preprocessed_contexts)]
    bm25 = BM25Okapi(tqdm(tokenized_wiki))

    return bm25

def make_hard_neg_ctxs(method, result_lst, passages, texts, answers, queries):
    topk_lst = []
    for i in tqdm(range(len(texts))):
        tokenized_query = tokenizer.tokenize(preprocess_retrieval(queries[i]))
        topk, neg_ctxs = 1, []
        while True:
            predict = method.get_top_n(tokenized_query, passages, n=topk)
            if answers[i] not in predict[-1]:  # 가장 마지막으로 들어온 예측 문장 안에 답이 없다면! -> hard_neg 찾음
                neg_ctxs.append(predict[-1])
                if len(neg_ctxs) == 3:
                    topk_lst.append(topk)
                    result_lst.append(neg_ctxs)
                    break
            topk += 1
        if i % 500 == 0:
            print("No worry! Things are working as planned.")
            print(f"{i}th example: max topk is {topk}\n Current lenth of result_lst {len(result_lst)}")
    return topk_lst

bm25 = prepare_bm25(contexts)
hard_neg_ctxs = []
topk_lst = make_hard_neg_ctxs(bm25, hard_neg_ctxs, contexts, train_context, train_answers, train_query)
topk_lst.sort(reverse=True)

print("Length of original train contexts :", len(train_context))
print("Length of hard negative contexts :", len(hard_neg_ctxs))
print("Most 10 k for making hard_neg_ctxs", topk_lst[:10])

if len(train_context) == len(hard_neg_ctxs):
    data["hard_neg_ctxs"] = hard_neg_ctxs
    data.to_csv("../data/negative_samples_hard.csv")

print("Finished making hard negative contexts!")