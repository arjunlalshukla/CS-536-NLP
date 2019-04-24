import pandas as pd
import numpy as np
from collections import Counter
import nltk
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from pymagnitude import *

data = pd.read_csv('data/questions.csv')
path = 'data/GoogleNews-vectors-negative300.magnitude'
vectors = Magnitude(path)

# helper functions
def sum_vec(words):
    return np.sum([vectors.query(w) for w in words if w in vectors], axis=0)

data.vecs1 = pd.Series([sum_vec(words) for words in data.question1])
data.vecs2 = pd.Series([sum_vec(words) for words in data.question2])

#initial data information
print("# of data points:", len(data))
print("# of word vectors:", len(vectors))
print('vector dimensions:', vectors.dim)
cnt = Counter(data.is_duplicate)
print("random baseline:", 1/len(cnt.keys()))
print("most common baseline:", max([cnt[k]/sum(cnt.values()) for k in cnt.keys()]))

#accuracy_score([0 if sentence_bleu(
#                        [str(data.question1[i]).split()],
#                        str(data.question2[i]).split()) 
#                < .5 else 1
#                for i in range(len(data))], data.is_duplicate)
