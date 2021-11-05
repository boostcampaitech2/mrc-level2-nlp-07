from datasets import load_from_disk, Features, Value, Sequence, DatasetDict, Dataset
import pandas as pd
import pickle
from elasticsearch import Elasticsearch
import os
from tqdm.auto import tqdm
import json
import re
import argparse


def preprocess_retrieval(corpus):
    corpus = corpus.replace("\\n", "")
    corpus = re.sub(f"[^- ㄱ-ㅎㅏ-ㅣ가-힣0-9a-zA-Zぁ-ゔァ-ヴー々〆〤一-龥]", " ", corpus)
    corpus = ' '.join(corpus.split())
    return corpus


def get_elasticsearch(preprocessed_contexts):
        
        os.system("service elasticsearch start")
        INDEX_NAME = "wiki_index"

        INDEX_SETTINGS = {"settings" : {"index":{"analysis":{"analyzer":{"korean":{"type":"custom",
                                                "tokenizer":"nori_tokenizer","filter": [ "shingle" ],}}}}},
        "mappings": {"properties" : {"context" : {"type" : "text","analyzer": "korean","search_analyzer": "korean"},}}}
        
        DOCS = {}
        for i in tqdm(range(len(preprocessed_contexts)), desc="preparing documents"):
            DOCS[i] = {'context':preprocessed_contexts[i]}
            
        try:
            es.transport.close()
        except:
            pass
        es = Elasticsearch(timeout=30, max_retries=10, retry_in_timeout=True)
        
        if es.indices.exists(INDEX_NAME):
            es.indices.delete(index=INDEX_NAME)
        es.indices.create(index=INDEX_NAME, body=INDEX_SETTINGS)
        
        for doc_id, doc in tqdm(DOCS.items(), desc="ES training..!"):
            es.index(index=INDEX_NAME,  id=doc_id, body=doc)

        return es


def get_relevent_elasticsearch(es, query, k=15):
    mod_query = preprocess_retrieval(query)
    try:
        res = es.search(index="wiki_index", q=mod_query, size=k)
    except:
        mod_q = mod_query.replace("%", " ").replace("-", " ")
        res = es.search(index="wiki_index", q=mod_q, size=k)
    
    doc_scores = [float(res['hits']['hits'][idx]['_score']) for idx in range(k)]
    doc_indices = [int(res['hits']['hits'][idx]['_id']) for idx in range(k)]
    return doc_scores, doc_indices


def process_long_context(dataset, es, contexts, num_hard, mode="original"):
    train_questions = list(dataset['question'])
    train_contexts = list(dataset['context'])

    hard_context = []
    for idx in tqdm(range(len(train_questions)), desc="Processing Long-context.."):
        text_set = []
        text_set.append(train_contexts[idx])
        _, doc_indices = get_relevent_elasticsearch(es, train_questions[idx])
        if mode != "original":
            doc_indices = doc_indices[::-1]
        for index in doc_indices:
            if contexts[index] == train_contexts[idx]:
                continue
            text_set.append(contexts[index])
            if len(text_set) == num_hard+1:
                break
        hard_context.append(' '.join(text_set))

    dataset['context'] = hard_context
    return dataset


def main(arg):
    assert os.path.exists(arg.concat_file), "Path of file to concat is wrong!"
    if arg.concat_file is not None:
        aug_df = pd.read_csv(arg.concat_file)

    with open(os.path.join("../data/", "wikipedia_documents.json"), "r", encoding="utf-8") as f:
        wiki = json.load(f)

    contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
    preprocessed_contexts = [preprocess_retrieval(corpus) for corpus in tqdm(contexts)]

    es = get_elasticsearch(preprocessed_contexts=preprocessed_contexts)

    train_dataset = pd.DataFrame(load_from_disk('../data/train_dataset/')['train'])
    train_dataset = process_long_context(train_dataset, es, contexts, arg.num_hard)

    if arg.concat_file is not None:
        try:
            aug_df['answers'] = [eval(ans) for ans in list(aug_df['answers'])]
        except:
            pass
        aug_df = process_long_context(aug_df, es, contexts, arg.num_hard, mode="aug")
    try:
        train_dataset['answers'] = [eval(ans) for ans in list(train_dataset['answers'])]
    except:
        print("Answers Format Correct")
        pass

    features = Features({'answers': Sequence(feature={'text': Value(dtype='string', id=None), 'answer_start': Value(dtype='int64', id=None)}, length=-1, id=None),
                            'context': Value(dtype='string', id=None),
                            'id': Value(dtype='string', id=None),
                            'question': Value(dtype='string', id=None)})

    if arg.concat_file is not None:
        train_df = pd.concat([train_dataset, aug_df, train_dataset])
    else:
        train_df = train_dataset
    valid_df = pd.DataFrame(load_from_disk("../data/train_dataset/")['validation'])

    dataset = DatasetDict({'train': Dataset.from_pandas(train_df, features=features),
                            'validation': Dataset.from_pandas(valid_df, features=features)})

    if arg.concat_file is not None:
        save_dir = f"../data/train_aug_hard{arg.num_hard}.pkl"
    else:
        save_dir = f"../data/train_hard{arg.num_hard}.pkl"
    file = open(save_dir, "wb")
    pickle.dump(dataset, file)
    file.close()
    print(f"New file save to {save_dir} !!")

    if arg.concat_file is not None:
        print(f"The ratio of original:augmentation = 2:1")
        print(f"If you set num_train_epochs = 1, model will be trained 3 epochs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_hard", type=int, help="Number of contexts to concat", default=5)
    parser.add_argument("--concat_file", type=str, help="Path of file to concat if exists", default=None)
    arg = parser.parse_args()
    main(arg)
