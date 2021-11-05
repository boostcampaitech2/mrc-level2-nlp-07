### 주의 : get_relevent_elasticsearch() 에서 k의 개수가 batch size의 배수가 돼야 오류가 안뜨는 것으로 추정
### elasticsearch 생성 5분 + inference 시간은 위의 k에 비례해서 늘어남
import json
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pprint import pprint
import re

import time
import os
import argparse
from elasticsearch import Elasticsearch
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from contextlib import contextmanager

from datasets import Dataset, load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    BertModel, BertPreTrainedModel,
    AdamW, get_linear_schedule_with_warmup,
    TrainingArguments,
)
from rank_bm25 import BM25Okapi
from preprocess import preprocess_retrieval

# 난수 고정
def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    random.seed(random_seed)
    np.random.seed(random_seed)

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")
    

class DenseRetrieval:
    def __init__(self,
        args,
        dataset,
        num_neg,
        tokenizer,
        p_encoder,
        q_encoder,
        mode,
        test_batch_size=4,
        wiki_path="wikipedia_documents.json",
    ):
        """
        학습과 추론에 사용될 여러 셋업을 마쳐봅시다.
        """

        self.args = args
        self.dataset = dataset
        self.num_neg = num_neg

        self.tokenizer = tokenizer
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder
        self.contexts = None

        self.mode = mode
        # self.passage_dataloader = None
        print(f"Mode : {self.mode}")
        self.p_with_neg = None
        if self.mode == "eval":
            with open(os.path.join("../data/", wiki_path), "r", encoding="utf-8") as f:
                wiki = json.load(f)

            self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
            print("Preprocessing Wiki Data..")
            self.preprocessed_contexts = [preprocess_retrieval(corpus) for corpus in tqdm(self.contexts)]
            os.system("service elasticsearch start")
            self.get_elasticsearch()
            # self.get_test_dataloader(self.tokenizer)
        if self.mode == "train":
            self.prepare_in_batch_negative(num_neg=num_neg)
    
    def get_test_dataloader(self, contexts, tokenizer):

        valid_seqs = tokenizer(
            contexts,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        passage_dataset = TensorDataset(
            valid_seqs["input_ids"],
            valid_seqs["attention_mask"],
            valid_seqs["token_type_ids"]
        )
        passage_dataloader = DataLoader(
            passage_dataset,
            batch_size=self.args.per_device_train_batch_size
        )
        return passage_dataloader
    
    def get_elasticsearch(self):

        self.es = Elasticsearch('localhost:9200', timeout=30, max_retries=10, retry_in_timeout=True)
        INDEX_NAME = "wiki_index"
        if self.es.indices.exists(INDEX_NAME):  # host에 이미 ES index 생성된 경우 -> 그대로 사용
            return 
        else:

            INDEX_SETTINGS = {"settings" : {"index":{"analysis":{"analyzer":{"korean":{"type":"custom",
                                                    "tokenizer":"nori_tokenizer","filter": [ "shingle" ]}}}}},
            "mappings": {"properties" : {"context" : {"type" : "text","analyzer": "korean","search_analyzer": "korean"},}}}
            
            DOCS = {}
            for i in tqdm(range(len(self.preprocessed_contexts)), desc="preparing documents"):
                DOCS[i] = {'context':self.preprocessed_contexts[i]}
                
            try:
                self.es.transport.close()
            except:
                pass
            self.es = Elasticsearch(timeout=30, max_retries=10, retry_in_timeout=True) 
            
            if self.es.indices.exists(INDEX_NAME):
                self.es.indices.delete(index=INDEX_NAME)
            self.es.indices.create(index=INDEX_NAME, body=INDEX_SETTINGS)
            
            for doc_id, doc in tqdm(DOCS.items(), desc="ES training..!"):
                self.es.index(index=INDEX_NAME,  id=doc_id, body=doc)


    def get_relevent_elasticsearch(self, query, k=20):
        mod_query = preprocess_retrieval(query)
        try:
            res = self.es.search(index="wiki_index", q=mod_query, size=k)
        except:
            mod_q = mod_query.replace("%", " ").replace("-", " ")
            res = self.es.search(index="wiki_index", q=mod_q, size=k)
        
        doc_scores = [float(res['hits']['hits'][idx]['_score']) for idx in range(k)]
        doc_indices = [int(res['hits']['hits'][idx]['_id']) for idx in range(k)]
        return doc_scores, doc_indices


    def prepare_in_batch_negative(self,
        dataset=None,
        num_neg=2,
        tokenizer=None
    ):

        if dataset is None:
            dataset = self.dataset

        if tokenizer is None:
            tokenizer = self.tokenizer

        # 1. In-Batch-Negative 만들기 (gold negative + hard negative)
        # CORPUS를 np.array로 변환해줍니다.
        print("Preparing In-batch Negative Samples")
        corpus = np.array(list(set([example for example in dataset["context"]])))
        p_with_neg = []

        for idx, c in enumerate(dataset["context"]):
            while True:
                neg_idxs = np.random.randint(len(corpus), size=num_neg)  # gold 방식!
                hard_neg_idx = random.randint(0, len(eval(dataset.loc[idx]["all_hard_neg"]))-1)  # hard_neg도 random 추출

                if not c in corpus[neg_idxs]:
                    p_neg = corpus[neg_idxs]

                    p_with_neg.append(c)
                    hard_neg = eval(dataset.loc[idx]["all_hard_neg"])[hard_neg_idx]
                    p_with_neg.append(hard_neg)
                    p_with_neg.extend(p_neg)
                    break
        
        questions = dataset["question"].tolist()


        # 2. (Question, Passage) 데이터셋 만들어주기
        q_seqs = tokenizer(
            questions,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        p_seqs = tokenizer(
            p_with_neg,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        max_len = p_seqs["input_ids"].size(-1)
        p_seqs["input_ids"] = p_seqs["input_ids"].view(-1, num_neg+2, max_len)
        p_seqs["attention_mask"] = p_seqs["attention_mask"].view(-1, num_neg+2, max_len)
        p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(-1, num_neg+2, max_len)

        train_dataset = TensorDataset(
            p_seqs["input_ids"], p_seqs["attention_mask"], p_seqs["token_type_ids"], 
            q_seqs["input_ids"], q_seqs["attention_mask"], q_seqs["token_type_ids"]
        )

        self.train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.args.per_device_train_batch_size
        )

        print(f"batch size : {self.args.per_device_train_batch_size}")


    def train(self, args=None):
        if args is None:
            args = self.args
        batch_size = args.per_device_train_batch_size

        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
            {"params": [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            {"params": [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
            {"params": [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon
        )
        t_total = len(self.train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=t_total
        )

        # Start training!
        global_step = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")
        # for _ in range(int(args.num_train_epochs)):
        for _ in train_iterator:

            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:

                    self.p_encoder.train()
                    self.q_encoder.train()
            
                    targets = torch.zeros(batch_size).long() # positive example은 전부 첫 번째에 위치하므로
                    targets = targets.to(args.device)

                    p_inputs = {
                        "input_ids": batch[0].view(batch_size * (self.num_neg + 2), -1).to(args.device),
                        "attention_mask": batch[1].view(batch_size * (self.num_neg + 2), -1).to(args.device),
                        "token_type_ids": batch[2].view(batch_size * (self.num_neg + 2), -1).to(args.device)
                    }
            
                    q_inputs = {
                        "input_ids": batch[3].to(args.device),
                        "attention_mask": batch[4].to(args.device),
                        "token_type_ids": batch[5].to(args.device)
                    }

                    # (batch_size*(num_neg+1), emb_dim)
                    p_outputs = self.p_encoder(**p_inputs)
                    # (batch_size*, emb_dim)
                    q_outputs = self.q_encoder(**q_inputs)

                    # Calculate similarity score & loss
                    p_outputs = torch.transpose(p_outputs.view(batch_size, self.num_neg+2, -1), 1, 2)
                    q_outputs = q_outputs.view(batch_size, 1, -1)

                    sim_scores = torch.bmm(q_outputs, p_outputs).squeeze()  #(batch_size, num_neg + 1)
                    sim_scores = sim_scores.view(batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets)
                    tepoch.set_postfix(loss=f"{str(loss.item())}")

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.p_encoder.zero_grad()
                    self.q_encoder.zero_grad()

                    global_step += 1

                    torch.cuda.empty_cache()

                    del p_inputs, q_inputs


    def get_relevant_doc(self,
        query,
        k=10,
        args=None,
        p_encoder=None,
        q_encoder=None
    ):
    
        if args is None:
            args = self.args

        if p_encoder is None:
            p_encoder = self.p_encoder

        if q_encoder is None:
            q_encoder = self.q_encoder

        scaler = MinMaxScaler()  # 정규화 
        ALPHA = 0.75
        es_scores, es_indices = self.get_relevent_elasticsearch(query, k=100)
        # es_scores = np.array(es_scores).reshape(-1, 1)  
        # es_scores = scaler.fit_transform(es_scores).flatten().tolist()  # es_scores 정규화

        cadidates = [self.contexts[idx] for idx in es_indices]
        mapping = {i:v for i, v in enumerate(es_indices)}
        passage_dataloader = self.get_test_dataloader(cadidates, self.tokenizer)
        es_tops = [[index, score] for index, score in zip(es_indices, es_scores)]

        with torch.no_grad():
            p_encoder.eval()
            q_encoder.eval()

            q_seqs_val = self.tokenizer(
                [query],
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(args.device)
            q_emb = q_encoder(**q_seqs_val).to("cpu")  # (num_query=1, emb_dim)

            p_embs = []
            for batch in passage_dataloader:

                batch = tuple(t.to(args.device) for t in batch)
                p_inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2]
                }
                p_emb = p_encoder(**p_inputs).to("cpu")
                p_embs.append(p_emb)

        # (num_passage, emb_dim)
        p_embs = torch.stack(
            p_embs, dim=0
        ).view(len(passage_dataloader.dataset), -1)

        dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
        dot_prod_scores = dot_prod_scores.squeeze().tolist()
        scores = [dot_prod_scores[idx] for idx in rank]

        # scores = np.array(scores).reshape(-1, 1)  
        # scores = scaler.fit_transform(scores).flatten().tolist()  # dense_scores 정규화

        rank = rank.tolist()
        dense_tops = [[mapping[index], score] for index, score in zip(rank, scores)]

        for i in range(len(es_tops)):
            for j in range(len(dense_tops)):
                if es_tops[i][0] == dense_tops[j][0]:
                    es_tops[i][1] += ALPHA*dense_tops[j][1]

        final_results = sorted(es_tops, key=lambda x: -x[-1])
        final_indices = [x[0] for x in final_results][:k]
        final_scores = [x[1] for x in final_results][:k]

        return final_scores, final_indices

    def get_relevant_doc_bulk(self, queries, k=10):

        doc_scores, doc_indices = [], []
        for query in queries:
            doc_score, doc_index = self.get_relevant_doc(query, k=k)
            doc_scores.append(doc_score)
            doc_indices.append(doc_index)
        return doc_scores, doc_indices

    
    def retrieve(
        self, query_or_dataset, topk=10):

        assert self.es is not None, "get_elasticsearch() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Golden retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context_id": doc_indices[idx],
                    "context_score": doc_scores[idx],
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]                    
                    ),
                    # "context": [self.contexts[pid] for pid in doc_indices[idx]] 
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas
    
    
class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()
      
    def forward(
            self,
            input_ids, 
            attention_mask=None,
            token_type_ids=None
        ): 

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs[1]
        return pooled_output
    
def main(arg):
    
    set_seed(42) # magic number :)
    print ("PyTorch version:[%s]."%(torch.__version__))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print ("device:[%s]."%(device))
    
    data_path = "../data/"

    if not os.path.exists('./dense_encoder'):
        os.mkdir('./dense_encoder')

    output_path = "./dense_encoder/" + arg.output_dir

    assert arg.mode.lower()=="train" or arg.mode.lower()=="eval", "Set Retrieval Mode : [train] or [eval]"
    
    if arg.mode.lower() == "train":
        train_path = "negative_samples_all.csv"
        train_dataset = pd.read_csv(data_path + train_path, engine="python")
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        args = TrainingArguments(
            output_dir=output_path,
            evaluation_strategy="epoch",
            learning_rate=arg.learning_rate,
            per_device_train_batch_size=arg.batch_size,
            per_device_eval_batch_size=arg.batch_size,
            num_train_epochs=arg.epoch,
            weight_decay=0.01
        )
        model_checkpoint = arg.model_name
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        p_encoder = BertEncoder.from_pretrained(model_checkpoint).to(args.device)
        q_encoder = BertEncoder.from_pretrained(model_checkpoint).to(args.device)

        retriever = DenseRetrieval(
            args=args,
            dataset=train_dataset,
            num_neg=arg.num_neg,
            tokenizer=tokenizer,
            p_encoder=p_encoder,
            q_encoder=q_encoder,
            mode=arg.mode.lower(),
        )
        retriever.train()
        
        torch.save(p_encoder, os.path.join(output_path, 'p_encoder.pt'))
        torch.save(q_encoder, os.path.join(output_path, 'q_encoder.pt'))
        print("Models Saved!")
        
        for i in range(5):
            query = train_dataset['question'][i]
            _, indices = retriever.get_relevant_doc(query=query, k=3)

            print(f"[Search Query] {query}\n")

            for i, idx in enumerate(indices):
                print(f"Top-{i + 1}th Passage (Index {idx})")
                pprint(retriever.contexts[idx]) 
            
    elif arg.mode.lower() == "eval":
        train_path = "train_dataset/"  # eval 시
        test_dataset = load_from_disk(data_path + train_path)["validation"]
        print(test_dataset['question'][0])
        print(test_dataset)
        
        assert os.path.exists(os.path.join(output_path, 'p_encoder.pt')) and os.path.exists(os.path.join(output_path, 'q_encoder.pt')), "Train and Load Models First!!"
        p_encoder = torch.load(os.path.join(output_path, 'p_encoder.pt')).to(device)
        q_encoder = torch.load(os.path.join(output_path, 'q_encoder.pt')).to(device)
        p_encoder.eval()
        q_encoder.eval()
        
        args = TrainingArguments(
            output_dir=os.path.join(output_path),
            evaluation_strategy="epoch",
            learning_rate=arg.learning_rate,
            # per_device_train_batch_size=1,
            # per_device_eval_batch_size=1,
            per_device_train_batch_size=arg.batch_size,
            per_device_eval_batch_size=arg.batch_size,
            num_train_epochs=arg.epoch,
            weight_decay=0.01
        )
        model_checkpoint = arg.model_name
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        
        retriever = DenseRetrieval(
            args=args,
            dataset=test_dataset,
            num_neg=arg.num_neg,
            tokenizer=tokenizer,
            p_encoder=p_encoder,
            q_encoder=q_encoder,
            mode=arg.mode.lower(),
        )
        
        print(f"Num Evaluation : {len(test_dataset)}")
        right_wrong = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
        k_lst = [1, 3, 5, 7, 9, 10, 15, 20]
        for i in tqdm(range(len(test_dataset['question']))):
            query = test_dataset['question'][i]
            if i % 50 == 0:
                print(query)
                print(test_dataset['context'][i])
            _, indices = retriever.get_relevant_doc(query=query, k=arg.topk)
            predict = []
            for k, idx in enumerate(indices):
                predict.append(retriever.contexts[idx])
                if i % 50 == 0:
                    print("-"*100)
                    print(f"Top-{k+1} predict")
                    print(retriever.contexts[idx])

            for k_idx in range(len(k_lst)):
                k = k_lst[k_idx]
                if test_dataset['context'][i] in predict[:k]:
                # if test_dataset['context'].tolist()[i] in predict:  # GPT-2 generated data
                    right_wrong[k_idx][0] += 1  # right
                else:
                    right_wrong[k_idx][1] += 1  # wrong
            
        print(output_path)
        for k_idx in range(len(k_lst)):
            right = right_wrong[k_idx][0]
            total = sum(right_wrong[k_idx])
            score = 100 * right / total
            print(f"Top-{k_lst[k_idx]} Acc. : {score:.2f}%")            
            
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="[train] or [eval]", default="train")
    parser.add_argument("--model_name", type=str, help="model name or path", default="klue/bert-base")
    parser.add_argument("--batch_size", type=int, help="per device batch size", default=4)
    parser.add_argument("--epoch", type=int, help="num train epochs", default=50)
    parser.add_argument("--learning_rate", type=float, help="train learning rate", default=1e-5)
    parser.add_argument("--num_neg", type=int, help="num negative sample per query", default=2)
    parser.add_argument("--output_dir", type=str, help="model save directory", default="dense_retrieval")
    parser.add_argument("--topk", type=int, help="num of Top-K for evaluation", default=10)
    arg = parser.parse_args()
    main(arg)