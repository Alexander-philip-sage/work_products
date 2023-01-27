import os
import json
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import hstack

from sklearn.model_selection import train_test_split, ShuffleSplit, learning_curve, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from gensim.parsing.preprocessing import preprocess_documents
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec


# ## Reads all wrong predictions / all models

files = glob.glob("output/*_wrong_predicts.csv")
dt = pd.concat([pd.read_csv(f) for f in files])


# ## Top 10 wrongly classified texts
print('Top 10 wrongly classified texts')
print(dt.groupby(['cik', 'text'], as_index=False)['year']   .count()   .rename(columns={'year':'n'})   .sort_values(['n'], ascending=False)   .head(10))


# ## Top 10 wrongly classified speakers
print('Top 10 wrongly classified speakers')
print(dt.groupby(['cik'], as_index=False)['year']   .count()   .rename(columns={'year':'n'})   .sort_values(['n'], ascending=False)   .head(10))


# ### Most difficult speakers and their texts
print('Most difficult speakers and their texts')
print(dt[dt['cik'].isin([310354, 719955, 58492, 78003, 816761, 316206])].groupby(['text'], as_index=False)['year']   .count()   .rename(columns={'year':'n'})   .sort_values(['n'], ascending=False))
