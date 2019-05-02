import pandas as pd
import numpy as np
from collections import *
import nltk
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from pymagnitude import *
import time
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('data/questions.csv')
data = data
path = 'data/GoogleNews-vectors-negative300.magnitude'
vectors = Magnitude(path)
tok = RegexpTokenizer('\w+')
data.question1 = pd.Series([tok.tokenize(str(s).lower()) for s in data.question1])
data.question2 = pd.Series([tok.tokenize(str(s).lower()) for s in data.question2])

