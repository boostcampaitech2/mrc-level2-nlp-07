import re

def preprocess_retrieval(corpus):
    corpus = corpus.replace("\\n", "")
    corpus = re.sub(f"[^- ㄱ-ㅎㅏ-ㅣ가-힣0-9a-zA-Zぁ-ゔァ-ヴー々〆〤一-龥]", " ", corpus)
    corpus = ' '.join(corpus.split())
    return corpus
