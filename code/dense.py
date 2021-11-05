import json
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pprint import pprint
import os
import argparse
import pickle

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from datasets import load_dataset, load_from_disk
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
    

class DenseRetrieval:
    def __init__(self,
        args,
        dataset,
        num_neg,
        tokenizer,
        p_encoder,
        q_encoder,
        mode
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

        self.mode = mode
        print(f"Mode : {self.mode}")
        self.p_with_neg = None
        if self.mode == "train":
            self.prepare_bm25()
        self.prepare_in_batch_negative(num_neg=num_neg)

    def prepare_bm25(self, dataset=None, model_name="monologg/koelectra-base-v3-discriminator"):
        if dataset is None:
            dataset = self.dataset
        
        contexts = list(set([example for example in dataset["context"]]))
        print("Tokenizing & Creating BM25 started!")
        preprocessed_contexts = [preprocess_retrieval(corpus) for corpus in tqdm(contexts)]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenized_wiki = [tokenizer.tokenize(corpus) for corpus in tqdm(preprocessed_contexts)]
        bm25 = BM25Okapi(tqdm(tokenized_wiki))

        self.p_with_neg = []
        print("Negative Samples...")
        for c in tqdm(dataset['context']):
            tokenized = tokenizer.tokenize(preprocess_retrieval(c))
            results = bm25.get_scores(tokenized)
            sorted_result = np.argsort(results)[::-1]
            doc_indices = sorted_result.tolist()

            self.p_with_neg.append(c)
            addition = []
            for idx in doc_indices:
                if c != dataset['context'][idx] and dataset['context'][idx] not in addition:
                    addition.append(dataset['context'][idx])
                if len(addition) == self.num_neg:
                    self.p_with_neg.extend(addition)
                    break
        print(f"Total Length : {len(self.p_with_neg)}")
        print(f"Total Positive Passages : {len(self.p_with_neg)//(self.num_neg)+1}")

    def prepare_in_batch_negative(self,
        dataset=None,
        num_neg=2,
        tokenizer=None
    ):

        if dataset is None:
            dataset = self.dataset

        if tokenizer is None:
            tokenizer = self.tokenizer

        # prepare_bm25를 통해 만든 p_with_neg 불러오기
        if self.mode == 'train':
            p_with_neg = self.p_with_neg
                
        else:
            p_with_neg = dataset['context']

        # 2. (Question, Passage) 데이터셋 만들어주기
        q_seqs = tokenizer(
            dataset["question"],
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
        if self.mode == 'train':
            p_seqs["input_ids"] = p_seqs["input_ids"].view(-1, num_neg+1, max_len)
            p_seqs["attention_mask"] = p_seqs["attention_mask"].view(-1, num_neg+1, max_len)
            p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(-1, num_neg+1, max_len)
        else:
            p_seqs["input_ids"] = p_seqs["input_ids"].view(-1, 1, max_len)
            p_seqs["attention_mask"] = p_seqs["attention_mask"].view(-1, 1, max_len)
            p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(-1, 1, max_len)

        train_dataset = TensorDataset(
            p_seqs["input_ids"], p_seqs["attention_mask"], p_seqs["token_type_ids"], 
            q_seqs["input_ids"], q_seqs["attention_mask"], q_seqs["token_type_ids"]
        )

        self.train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.args.per_device_train_batch_size
        )

        valid_seqs = tokenizer(
            dataset["context"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        passage_dataset = TensorDataset(
            valid_seqs["input_ids"],
            valid_seqs["attention_mask"],
            valid_seqs["token_type_ids"]
        )
        self.passage_dataloader = DataLoader(
            passage_dataset,
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
                        "input_ids": batch[0].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        "attention_mask": batch[1].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        "token_type_ids": batch[2].view(batch_size * (self.num_neg + 1), -1).to(args.device)
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
                    p_outputs = torch.transpose(p_outputs.view(batch_size, self.num_neg+1, -1), 1, 2)
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
        k=1,
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
            for batch in self.passage_dataloader:

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
        ).view(len(self.passage_dataloader.dataset), -1)

        dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()

        ###### 디버깅 코드 ########
        # print(dot_prod_scores.shape)
        # print(dot_prod_scores)
        # high_scores = []
        # for i in rank[:k]:
        #     high_scores.append(dot_prod_scores[i])

        # print(high_scores)

        return rank[:k]
    
    
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
    train_path = "train_dataset/"
    
    assert arg.mode.lower()=="train" or arg.mode.lower()=="eval", "Set Retrieval Mode : [train] or [eval]"
    
    if arg.mode.lower() == "train":
        train_dataset = load_from_disk(data_path + train_path)["train"]

        args = TrainingArguments(
            output_dir=os.path.join(data_path, arg.output_dir),
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
        
        torch.save(p_encoder, os.path.join(data_path, 'p_encoder.pt'))
        torch.save(q_encoder, os.path.join(data_path, 'q_encoder.pt'))
        print("Models Saved!")
        
        for i in range(5):
            query = train_dataset['question'][i]
            results = retriever.get_relevant_doc(query=query, k=3)

            print(f"[Search Query] {query}\n")

            indices = results.tolist()
            for i, idx in enumerate(indices):
                print(f"Top-{i + 1}th Passage (Index {idx})")
                pprint(retriever.dataset["context"][idx]) 
            
    elif arg.mode.lower() == "eval":
        test_dataset = load_from_disk(data_path + train_path)["validation"]
        
        assert os.path.exists(os.path.join(data_path, 'p_encoder.pt')) and os.path.exists(os.path.join(data_path, 'q_encoder.pt')), "Train and Load Models First!!"
        p_encoder = torch.load(os.path.join(data_path, 'p_encoder.pt')).to(device)
        q_encoder = torch.load(os.path.join(data_path, 'q_encoder.pt')).to(device)
        # p_encoder.to(device)
        # q_encoder.to(device)
        p_encoder.eval()
        q_encoder.eval()
        
        args = TrainingArguments(
            output_dir=os.path.join(data_path, arg.output_dir),
            evaluation_strategy="epoch",
            learning_rate=arg.learning_rate,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            # per_device_train_batch_size=arg.batch_size,
            # per_device_eval_batch_size=arg.batch_size,
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
        right, wrong = 0, 0
        for i in tqdm(range(len(test_dataset))):
            query = test_dataset['question'][i]
            print(query)
            print(test_dataset['context'][i])
            results = retriever.get_relevant_doc(query=query, k=arg.topk)
            indices = results.tolist()
            predict = []
            for k, idx in enumerate(indices):
                predict.append(retriever.dataset["context"][idx])
                print("-"*100)
                print(f"Top-{k+1} predict")
                print(retriever.dataset["context"][idx])
            if test_dataset['context'][i] in predict:
                right += 1
            else:
                wrong += 1
            
        print(f"Top-{arg.topk} Acc. : {100*right/(right+wrong):.2f}%")            
            
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="[train] or [eval]", default="train")
    parser.add_argument("--model_name", type=str, help="model name or path", default="klue/bert-base")
    parser.add_argument("--batch_size", type=int, help="per device batch size", default=4)
    parser.add_argument("--epoch", type=int, help="num train epochs", default=50)
    parser.add_argument("--learning_rate", type=float, help="train learning rate", default=1e-5)
    parser.add_argument("--num_neg", type=int, help="num negative sample per query", default=2)
    parser.add_argument("--output_dir", type=str, help="model save directory", default="dense_retrieval")
    parser.add_argument("--topk", type=int, help="num of Top-K for evaluation", default=3)
    arg = parser.parse_args()
    main(arg)

# Negative Sample 만드는 시간이 매우 오래 걸림