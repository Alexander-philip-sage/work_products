#**************************************************************************************************
# BOARD LEADERSHIP VARIABLE
# FILE: SAMPLE SCORING
#**************************************************************************************************

###################################################################################################
# IMPORTS
###################################################################################################

import os
import json
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

###################################################################################################
# GLOBAL PARAMETERS
###################################################################################################

random_state=2020       # reproducibility seed

test_size=0.20          # fraction of hold-out data
cv_folds=5              # number of folds for cross-validation
cv_test=0.20            # fraction of test data in cross-validation
max_iter=1e3            # maximum number of iterations allowed (CV and TF-IDF)
max_features=int(1e6)   # maximum number of features after vectorization (CV and TF-IDF)
score='accuracy'        # Main metric for modeling

d2v_size=200            # vector size
d2v_window=5            # learning window
d2v_mincount=1          # minimal word frequency allowed
d2v_epochs=100          # maximum number of iterations

w2v_size=200            # vector size
w2v_window=5            # learning window
w2v_mincount=1          # minimal word frequency allowed

lda_ntopics=100         # Number of LDA topics

###################################################################################################
# SCORING FUNCTION
###################################################################################################

def scoring(target, model, model_features, ngram_range, fname, top=None):
    
    """
    Utility function to score data with pre-built models
    
    `target`: Target variable name
    `model`: type of the model 'lg' or 'rf'
    `model_features`: list of used features, incl. 'cv', 'tfidf', 'word2vec', 'doc2vec', 'lda'
    `nrgam_range`: N-grams range
    `fname`: File name for data
    `top`: How many samples to score (if None - all rows are scored)
    
    Stores generated output to the disk and returns data.frame with results
    """
    
    # (1) Reads data
    if not os.path.exists(fname):
        print('Data file "%s" doesn\'t exist!' % fname)
        return
    dt = pd.read_csv(fname).head(top)
    X = dt['text']
    
    # (2) Loads model
    model_path = 'output/%s-%s-%s-ngram%02d_model.pickle' % (target, model, '_'.join(model_features), max(ngram_range))
    if not os.path.exists(model_path):
        print('Model "%s" doesn\'t exist...' % model_path)
        return None
    else:
        with open(model_path, 'rb') as f:
            scorer = pickle.load(f)
    
    # (3) Loads features transformers & create features
    X_ = None

    for feature in model_features:
        if feature == 'cv':
            with open('transformers/%s_cv_ngram_%02d.pickle' % (target, max(ngram_range)), 'rb') as f:
                vectorizer_cv = pickle.load(f)
            X_cv = vectorizer_cv.transform(X)
            X_ = hstack([X_, X_cv])
            
        elif feature == 'tfidf':
            with open('transformers/%s_tfidf_ngram_%02d.pickle' % (target, max(ngram_range)), 'rb') as f:
                vectorizer_tfidf = pickle.load(f)
            X_tfidf = vectorizer_tfidf.transform(X)
            X_ = hstack([X_, X_tfidf])
    
        elif feature == 'doc2vec':
            with open('transformers/%s_doc2vec_dm_ngram_%02d.pickle' % (target, max(ngram_range)), 'rb') as f:
                model_dm = pickle.load(f)
            with open('transformers/%s_doc2vec_dbow_ngram_%02d.pickle' % (target, max(ngram_range)), 'rb') as f:
                model_dbow = pickle.load(f)
            
            corpus_test = preprocess_documents(list(X))
            X_doc2vec = np.array([np.hstack((model_dm.infer_vector(i), 
                                             model_dbow.infer_vector(i))) for i in corpus_test])
            X_ = hstack([X_, X_doc2vec])
            
        elif feature == 'word2vec':
            with open('transformers/%s_word2vec_cbow_ngram_%02d.pickle' % (target, max(ngram_range)), 'rb') as f:
                model_cbow = pickle.load(f)
            with open('transformers/%s_word2vec_skipgram_ngram_%02d.pickle' % (target, max(ngram_range)), 'rb') as f:
                model_skipgram = pickle.load(f)
            
            corpus_test = preprocess_documents(list(X))
            X_word2vec = np.empty(shape=(X.shape[0], 2*w2v_size), dtype='float32')
            
            for i in range(len(corpus_test)):
                test_text = [w for w in corpus_test[i] if w in model_cbow.wv.vocab]
                X_word2vec[i, ] = np.hstack(
                    (np.average(model_cbow.wv[test_text], axis=0),
                     np.average(model_skipgram.wv[test_text], axis=0)))
                
            X_ = hstack([X_, X_word2vec])
            
        elif feature == 'lda':
            with open('transformers/%s_lda_ngram_%02d.pickle' % (target, max(ngram_range)), 'rb') as f:
                lda = pickle.load(f)
            with open('transformers/%s_cv_ngram_%02d.pickle' % (target, max(ngram_range)), 'rb') as f:
                vectorizer_cv = pickle.load(f)
            X_cv = vectorizer_cv.transform(X)        
            X_lda = lda.transform(X_cv)
            X_ = hstack([X_, X_lda])
            
        else:
            raise 'Unknown model feature!'
    
    # (4) Scores
    pred = pd.DataFrame(np.column_stack((scorer.predict(X_), scorer.predict_proba(X_))),
                        columns=['predicted_class', 'prob0', 'prob1'])
    result = pd.concat([dt, pred], axis=1)
    result_path = 'scoring/%s' % os.path.basename(fname).replace('.csv', '-%s-%s-%s-ngram%02d-scoring.csv' % (target, model, '_'.join(model_features), max(ngram_range)))
    result.to_csv(result_path, index=False)
    print('Results are saved to "%s"' % result_path)

    return result

###################################################################################################
# SAMPLE SCORING
###################################################################################################

scoring(target = 'rbv',
        model = 'rf',
        model_features = ('cv','tfidf','word2vec',),
        ngram_range = (1, 2),
        fname = 'data/extracts-fullsample.csv',
        top=100)

# Scoring with top 3 models for each target variable

results = pd.read_csv('output/results.csv')
top3 = results.groupby(['target'])               .apply(lambda x: x.sort_values(by='accuracy_test', ascending=False)               .head(3))               .reset_index(drop=True)
print(top3)

for _, row in top3.iterrows():
    scoring(target=row['target'],
            model=row['model'],
            model_features=row['model_features'].split('_'),
            ngram_range=(1, row['ngram']),
            fname='data/extracts-fullsample.csv')

# END