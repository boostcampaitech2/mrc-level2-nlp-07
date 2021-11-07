# Boostcamp Machine Reading Comprehension Competition
## **Table of contents**

1. [Introduction](#introduction)
2. [Project Outline](#project-outline)
3. [Solution](#solution)
4. [How to Use](#how-to-use)

# 1. Introduction  
<br/>
<p align="center">
   <img src="https://user-images.githubusercontent.com/62708568/136650411-a9923f11-eb89-4832-8c86-89ee48c62f69.png" style="width:800px;"/>
</p>

<br/>


## ☕ 조지KLUE니

## **개요**

1. Introduction
2. Project Outline
3. Solution
4. How to Use

# 1. Introduction

[🔅 Members](https://www.notion.so/bcc26f407b22470a9cbcaa6a238b573f)

### 🔅 Contribution

`김보성` Modeling(MaskedLM with Bi-LSTM, MaskedLM with Autoencoder)•Reference searching•Paper implementation•Ensemble•github management

`김지후`  

`김혜수` Reference Searching•ElasticSearch config & Optimization•Data Processing•Sparse/Dense Retrieval•Re-ranking MRC outputs w/ Retrieval

`박이삭` Reference Searching•Github management

`이다곤` Data Processing•Generative MRC

`전미원` Data Preprocessing•Add Elastic Search into baseline•Re-ranking MRC outputs w/ Retrieval•Ensemble

`정두해` Data Exploration•Baseline Abstraction•Sparse/Dense Retriever•Reader Model Searching•Data Augmentation•MRC Hyperparameter Tuning•Pre/Postprocessing

# 2. Project Outline

- Task : Extractive-based MRC를 위한 ODQA 모델 구축
- Date : 2021.10.12 - 2021.11.04 (4 weeks)
- Description : **본 ODQA 대회에서 우리가 만들 모델은 two-stage**로 구성되어 있습니다. **첫 단계는 질문에 관련된 문서를 찾아주는 "retriever"** 단계이고, **다음으로는 관련된 문서를 읽고 적절한 답변을 찾거나 만들어주는 "reader"** 단계입니다. 두 가지 단계를 각각 구성하고 그것들을 적절히 통합하게 되면, 어려운 질문을 던져도 답변을 해주는 ODQA 시스템을 여러분들 손으로 직접 만들어보게 됩니다.
- Train : 3,952개
- Validation : 240개
- Test : 600개

### 🏆 Final Score

대회 사이트 : [AI stage](https://stages.ai/competitions/75/overview/description)

## **Hardware**

AI stage에서 제공한 server, GPU

- GPU: V100

# 3. Solution

### KEY POINT

- DPR 논문의 Gold 방식의 Dense Retriever 모델을 차용해 elasticsearch와 결합하여 retriever 모델 구현
- Data Augmentation을 통해 지문의 길이를 늘린 후 학습 데이터로 이용
- 대량의 한국어 데이터로 사전학습 되어 있는 klue/roberta-large 모델을 리더 모델로 사용

### Checklist

- [x]  EDA
- [x]  Data Preprocessing(`special character removal`, `getting answer spans' start position with special character tokens`)
- [x]  Data Augmentation(`Back translation`, `Question generation`)
- [x]  Data Postprocessing
- [x]  Experimental Logging (`WandB`)
- [x]  Retrieval (`dense -- FAISS,using simple dual-encoders`, `sparse -- TF-IDF,BM25,Elastic search`)
- [x]  Custom Model Architecture(`Roberta with BiLSTM`, `Roberta with Autoencoder`)
- [x]  Re-ranker (`changing scoring function using BERTserini`)
- [x]  Ensemble
- [ ]  K-fold cross validation
- [ ]  Shorten inference time when using elastic search

[Evaluation](https://www.notion.so/b3aac65c45924c378f0ec07f7b05a38a)

# 4. How to Use

## **Installation**

다음과 같은 명령어로 필요한 libraries를 다운 받습니다.

`pip install -r requirements.txt`

Elasticsearch 모듈 (출처 : [서중원 멘토님 깃허브](https://github.com/thejungwon/search-engine-tutorial))

`apt-get update && apt-get install -y gnupg2`

`wget -qO - [https://artifacts.elastic.co/GPG-KEY-elasticsearch](https://artifacts.elastic.co/GPG-KEY-elasticsearch) | apt-key add -`

`apt-get install apt-transport-https`

`echo "deb [https://artifacts.elastic.co/packages/7.x/apt](https://artifacts.elastic.co/packages/7.x/apt) stable main" | tee /etc/apt/sources.list.d/elastic-7.x.list`

`apt-get update && apt-get install elasticsearch`

`service elasticsearch start`

`cd /usr/share/elasticsearch`

`bin/elasticsearch-plugin install analysis-nori`

`service elasticsearch restart`

`pip install elasticsearch`

BM25 모듈

`pip install rank_bm25`

Google deep_translator 모듈

`pip install -U deep-translator`

## **Dataset**

파일: data/train_dataset/train, data/train_dataset/validation, data/test_dataset/validation 

## **Data Analysis**

파일: 

## **Data preprocessing**

파일: 

## **Modeling**

파일: train.py, inference.py, 

## **Ensemble**

파일: mixing_bowl.ipynb, mixing_bowl (1).ipynb

## Directory

```
.
├── mrc-level2-nlp-07
|    ├── code
│        ├── outputs
│        ├── dense_encoder
│        ├── retriever
|    ├── data
│        ├── train_dataset
|            ├── train
|            ├── validation
│        ├── test_dataset
|            ├── validation
|        ├── wikipedia_passages.json
```

- `code` 파일 안에는 각각 **data preprocessing** • **train** • **inference**가 가능한 라이브러리가 들어있습니다.
- `train.py`를 실행시키면 logs, results, best_model 폴더에 결과들이 저장됩니다.
- 사용자는 전체 코드를 내려받은 후, argument 옵션을 지정하여 개별 라이브러리 모델을 활용할 수 있습니다.
