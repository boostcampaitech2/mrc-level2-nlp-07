import re
from transformers import AutoTokenizer

commons = ['▁', '의', '을', '에', '이', '은', '는', '년', '를', '로', '가', '에서', '▁이', '으로', '한', '고', '과', '인', '도', '와', '월', '리', '지', '일', '사', '스', '▁수', '기', '다', '▁있다', '어', '했다', '시', '르', '하였다', '▁그', '자', '하는', '라', '해', '▁전', '하고', '이다', '부', '하여', '군', '▁1', '▁가', '▁사', '대']
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)

def preprocess_retrieval(corpus):
    corpus = corpus.replace("\\n", "")
    corpus = re.sub(f"[\"<>\[\].,?!\(\)\:#\|'\=-]", " ", corpus)
    corpus = ' '.join(corpus.split())
    return corpus

def tokenizer_filter(corpus):
    tokenized = tokenizer.tokenize(preprocess_retrieval(corpus))
    filtered = [token for token in tokenized if token not in commons]
    return filtered

def preprocess_mrc(corpus):
    corpus = corpus.replace(f"\n", "")
    corpus = ' '.join(corpus.split())
    return corpus