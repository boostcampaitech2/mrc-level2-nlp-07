import re

def preprocess_retrieval(corpus):
    corpus = corpus.replace(f"\n", "")
    corpus = ' '.join(corpus.split())
    corpus = re.sub(f"[\"<>\[\].,?!\(\)\:#\|'\=-]", "", corpus)
    return corpus


def preprocess_mrc():
    pass