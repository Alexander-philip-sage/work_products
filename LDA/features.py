#**************************************************************************************************
# BOARD LEADERSHIP VARIABLE
# FILE: FEATURE GENERATION
#**************************************************************************************************

###################################################################################################
# IMPORTS
###################################################################################################

import os
import json
import pickle
from typing import Iterable, List, Tuple
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
from sklearn.metrics import mean_squared_error

from gensim.parsing.preprocessing import preprocess_documents
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models import LdaMulticore
###################################################################################################
# GLOBAL PARAMETERS
###################################################################################################

random_state=2020       # reproducibility seed

test_size=0.20          # fraction of hold-out data
max_features=int(1e6)   # maximum number of features after vectorization (CV and TF-IDF)

d2v_size=200            # vector size
d2v_window=5            # learning window
d2v_mincount=1          # minimal word frequency allowed
d2v_epochs=100          # maximum number of iterations

w2v_size=200            # vector size
w2v_window=5            # learning window
w2v_mincount=1          # minimal word frequency allowed


###################################################################################################
# METHODOLOGY
###################################################################################################

# Steps in the analysis:
# 
# 1. Text pre-processing
# 1.1 Standard preprocessing
# 1.2 N-grams
#  
# 2. Feature generation
# 2.1 Word Counts
# 2.2 Term Frequency - Inverse Document Frequency (TF-IDF)
# 2.3 Word2Vec Embedding
# 2.4 Doc2Vec Embedding
# 2.5 Latent Dirichlet Allocation (LDA)
# 
# 3 Modeling
# 3.1 Train/test splitting
# 3.2 Cross-validation
# 3.3 Classification models
# 3.4 Hyper-parameters tuning
# 

###################################################################################################
# STEPS 1 & 2 (PREPROCESSING & FEATURE GENERATION)
###################################################################################################

def load_data(target, fname, subset = None):
    # Reads data for specified target variable
    df = pd.read_csv('data/%s' % fname)
    df = df[['cik', 'text', 'year', target]].drop_duplicates()
    if subset:
        assert subset > 0, "a subset of 0 means no data"
        df = df.iloc[:int(subset*df.shape[0])]
    
    # Train/test split
    y = df[target]
    X = df[['cik', 'year', 'text']]
    
    
    if not os.path.exists('features'):
        os.mkdir('features')

    return  train_test_split(X, y, test_size=test_size, random_state=random_state)


def pick_raw_data(y_train, X_train_, y_test, X_test_, target, ngram_range):
    # Saves target & X variables to disk
    with open('features/%s_train_target_ngram_%02d.pickle' % (target, max(ngram_range)), 'wb') as f:
        print("Storing '%s'..." % f.name)
        pickle.dump(y_train, f)
    
    with open('features/%s_train_X_ngram_%02d.pickle' % (target, max(ngram_range)), 'wb') as f:
        print("Storing '%s'..." % f.name)
        pickle.dump(X_train_, f)
        
    with open('features/%s_test_target_ngram_%02d.pickle' % (target, max(ngram_range)), 'wb') as f:
        print("Storing '%s'..." % f.name)
        pickle.dump(y_test, f)
    
    with open('features/%s_test_X_ngram_%02d.pickle' % (target, max(ngram_range)), 'wb') as f:
        print("Storing '%s'..." % f.name)
        pickle.dump(X_test_, f)

def count_vectorizer_features(X_train_,X_test_ , target, ngram_range):
    vectorizer_cv = CountVectorizer(stop_words='english',
                                    strip_accents='unicode', 
                                    ngram_range=ngram_range,
                                    max_features=max_features)
    
    X_train_cv = vectorizer_cv.fit_transform(X_train_)
    X_test_cv = vectorizer_cv.transform(X_test_)
    
    with open('features/%s_train_cv_ngram_%02d.pickle' % (target, max(ngram_range)), 'wb') as f:
        print("Storing '%s'..." % f.name)
        pickle.dump(X_train_cv, f)
        
    with open('features/%s_test_cv_ngram_%02d.pickle' % (target, max(ngram_range)), 'wb') as f:
        print("Storing '%s'..." % f.name)
        pickle.dump(X_test_cv, f)
        
    if not os.path.exists('transformers'):
        os.mkdir('transformers')


    with open('transformers/%s_cv_ngram_%02d.pickle' % (target, max(ngram_range)), 'wb') as f:
        print("Storing '%s'..." % f.name)
        pickle.dump(vectorizer_cv, f)    
    return X_train_cv, X_test_cv

def lda_features_sklearn(lda_ntopics, n_workers, X_train_cv, X_test_cv):
    lda = LDA(n_components=lda_ntopics,
              learning_method='batch',
              n_jobs=n_workers,
              random_state=random_state)
    
    X_train_lda = lda.fit_transform(X_train_cv)
    X_test_lda = lda.transform(X_test_cv)
    perplexity = lda.perplexity(X_test_cv)
    return perplexity, X_train_lda, X_test_lda, lda
    
def sklearn_cv_to_gensim(cv: Iterable[int]) -> List[List[Tuple[int]]]:
    """takes in the from sklearn.feature_extraction.text import CountVectorizer and 
    returns a format that Dictionary doc2bow from gensim would return
    """
    ret = []
    for text in cv:
        tmp = []
        for i, count in enumerate(text):
            #i is the word id, count is the frequency
            if count > 0:
                tmp.append((i,count))
        ret.append(tmp)
    return ret

def lda_features_gensim(lda_ntopics, n_workers, X_train_cv, X_test_cv):
    X_train_cv = sklearn_cv_to_gensim(X_train_cv.toarray())
    X_test_cv = sklearn_cv_to_gensim(X_test_cv.toarray())
 
    if n_workers == 1:
        lda = LdaModel(X_train_cv, num_topics=lda_ntopics, random_state=random_state)
    else:
        lda = LdaMulticore(X_train_cv, num_topics=lda_ntopics, random_state=random_state)

    X_train_lda = lda.inference(X_train_cv)
    X_test_lda = lda.inference(X_test_cv)
    perwordbound = np.float64(lda.log_perplexity(X_test_cv))
    perplexity = np.exp2(-perwordbound)
    return perplexity, X_train_lda[0], X_test_lda[0], lda

def lda_features(target: str, fname: str, ngram_range: Tuple[int], lda_ntopics: int, n_workers=os.cpu_count(), subset = None, module='gensim'):
    """
    Utility function to generates features out of text data with following parameters
    feature types created: LDA

    `target`: Target variable name
    `fname`: File name for data
    `ngram_range`: N-grams range
    `n_workers`: number of OS jobs to use for models training
    'subset': expects value 0 < subset <=1
                reduces the data by a factor of subset
    
    Stores generated features to the disk for re-use
    """
    X_train_, X_test_, y_train, y_test = load_data(target, fname, subset = subset)
    pick_raw_data(y_train, X_train_, y_test, X_test_, target, ngram_range)
    X_train_ = X_train_['text']
    X_test_ = X_test_['text'] 

    X_train_cv, X_test_cv = count_vectorizer_features(X_train_,X_test_ , target, ngram_range)

    # Latent Dirichlet Allocation (LDA) Vectorizer
    if module=='gensim':
        perplexity, X_train_lda, X_test_lda, lda = lda_features_gensim(lda_ntopics, n_workers, X_train_cv, X_test_cv)
    elif module=='sklearn':
        perplexity, X_train_lda, X_test_lda, lda = lda_features_sklearn(lda_ntopics, n_workers, X_train_cv, X_test_cv)
    else:
        raise ValueError("only accepts gensim or sklearn")

    
    with open('features/%s_train_lda_ngram_%02d_ntop_%03d.pickle' % (target, max(ngram_range), lda_ntopics), 'wb') as f:
        print("Storing '%s'..." % f.name)
        pickle.dump(X_train_lda, f)
        
    with open('features/%s_test_lda_ngram_%02d_ntop_%03d.pickle' % (target, max(ngram_range), lda_ntopics), 'wb') as f:
        print("Storing '%s'..." % f.name)
        pickle.dump(X_test_lda, f)
        
    with open('transformers/%s_%s_lda_ngram_%02d_ntop_%03d.pickle' % (target, module, max(ngram_range), lda_ntopics), 'wb') as f:
        print("Storing '%s'..." % f.name)
        pickle.dump(lda, f)
    
    return perplexity



def tf_idf_features(X_train_,X_test_ , target, ngram_range):
    vectorizer_tfidf = TfidfVectorizer(stop_words='english',
                                       strip_accents='unicode',
                                       ngram_range=ngram_range,
                                       max_features=max_features)
    
    X_train_tfidf = vectorizer_tfidf.fit_transform(X_train_)
    X_test_tfidf = vectorizer_tfidf.transform(X_test_)
    
    with open('features/%s_train_tfidf_ngram_%02d.pickle' % (target, max(ngram_range)), 'wb') as f:
        print("Storing '%s'..." % f.name)
        pickle.dump(X_train_tfidf, f)
        
    with open('features/%s_test_tfidf_ngram_%02d.pickle' % (target, max(ngram_range)), 'wb') as f:
        print("Storing '%s'..." % f.name)
        pickle.dump(X_test_tfidf, f)
        
    with open('transformers/%s_tfidf_ngram_%02d.pickle' % (target, max(ngram_range)), 'wb') as f:
        print("Storing '%s'..." % f.name)
        pickle.dump(vectorizer_tfidf, f)    

def doc_embedding_featues(corpus_train, corpus_test, n_workers, target, ngram_range):
    documents_train = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus_train)]
    #documents_test = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus_test)]
    
    # Combines distributed memory (DM) and distributed bag-of-words (DBOW) for better results
    model_dm = Doc2Vec(documents_train, 
                       dm=1,
                       vector_size=d2v_size,
                       window=d2v_window,
                       min_count=d2v_mincount,
                       epochs=d2v_epochs,
                       workers=n_workers,
                       seed=random_state)
    
    model_dbow = Doc2Vec(documents_train, 
                         dm=0, 
                         vector_size=d2v_size,
                         window=d2v_window,
                         min_count=d2v_mincount,
                         epochs=d2v_epochs,
                         workers=n_workers,
                         seed=random_state)
    
    X_train_doc2vec = np.hstack((model_dm.docvecs.vectors_docs, model_dbow.docvecs.vectors_docs))
    X_test_doc2vec = np.array([np.hstack((model_dm.infer_vector(i), 
                                          model_dbow.infer_vector(i))) for i in corpus_test])
    
    with open('features/%s_train_doc2vec_ngram_%02d.pickle' % (target, max(ngram_range)), 'wb') as f:
        print("Storing '%s'..." % f.name)
        pickle.dump(X_train_doc2vec, f)
        
    with open('features/%s_test_doc2vec_ngram_%02d.pickle' % (target, max(ngram_range)), 'wb') as f:
        print("Storing '%s'..." % f.name)
        pickle.dump(X_test_doc2vec, f)
        
    with open('transformers/%s_doc2vec_dm_ngram_%02d.pickle' % (target, max(ngram_range)), 'wb') as f:
        print("Storing '%s'..." % f.name)
        pickle.dump(model_dm, f)
    
    with open('transformers/%s_doc2vec_dbow_ngram_%02d.pickle' % (target, max(ngram_range)), 'wb') as f:
        print("Storing '%s'..." % f.name)
        pickle.dump(model_dbow, f)    

def word_embedding_features(corpus_train, corpus_test, X_train_, X_test_, n_workers, target, ngram_range):
    # CBOW model
    model_cbow = Word2Vec(corpus_train,
                          min_count=w2v_mincount,
                          size=w2v_size,
                          window=w2v_window,
                          sg=0,
                          workers=n_workers,
                          seed=random_state)
    
    # Skip-gram model
    model_skipgram = Word2Vec(corpus_train,
                              min_count=w2v_mincount,
                              size=w2v_size,
                              window=w2v_window,
                              sg=1,
                              workers=n_workers,
                              seed=random_state)
    
    # Averaging train vectors
    X_train_word2vec = np.empty(shape=(X_train_.shape[0], 2*w2v_size), dtype='float32')
    
    for i in range(len(corpus_train)):
        X_train_word2vec[i, ] = np.hstack(
            (np.average(model_cbow.wv[corpus_train[i]], axis=0),
             np.average(model_skipgram.wv[corpus_train[i]], axis=0)))
    
    # Averaging test vectors (skipping missing words)
    X_test_word2vec = np.empty(shape=(X_test_.shape[0], 2*w2v_size), dtype='float32')
    
    for i in range(len(corpus_test)):
        test_text = [w for w in corpus_test[i] if w in model_cbow.wv.vocab]
        X_test_word2vec[i, ] = np.hstack(
            (np.average(model_cbow.wv[test_text], axis=0),
             np.average(model_skipgram.wv[test_text], axis=0)))
    
    with open('features/%s_train_word2vec_ngram_%02d.pickle' % (target, max(ngram_range)), 'wb') as f:
        print("Storing '%s'..." % f.name)
        pickle.dump(X_train_word2vec, f)
        
    with open('features/%s_test_word2vec_ngram_%02d.pickle' % (target, max(ngram_range)), 'wb') as f:
        print("Storing '%s'..." % f.name)
        pickle.dump(X_test_word2vec, f)
        
    with open('transformers/%s_word2vec_cbow_ngram_%02d.pickle' % (target, max(ngram_range)), 'wb') as f:
        print("Storing '%s'..." % f.name)
        pickle.dump(model_cbow, f)
    
    with open('transformers/%s_word2vec_skipgram_ngram_%02d.pickle' % (target, max(ngram_range)), 'wb') as f:
        print("Storing '%s'..." % f.name)
        pickle.dump(model_skipgram, f)



def general_features(target, fname, ngram_range, n_workers=os.cpu_count(), subset = None):
    
    """
    Utility function to generates features out of text data with following parameters
    Files Created:
        feature types created: Word2Vec, Count, TF-IDF, Doc2Vec
        Saves raw organized data

    Parameters
        `target`: Target variable name
        `fname`: File name for data
        `ngram_range`: N-grams range
        `n_workers`: number of OS jobs to use for models training
        'subset': expects value 0 < subset <=1
                    reduces the data by a factor of subset
    
    Stores generated features to the disk for re-use
    """
    
    X_train_, X_test_, y_train, y_test = load_data(target, fname, subset = subset)
    pick_raw_data(y_train, X_train_, y_test, X_test_, target, ngram_range)
    X_train_ = X_train_['text']
    X_test_ = X_test_['text']        
    
    X_train_cv, X_test_cv = count_vectorizer_features(X_train_,X_test_ , target, ngram_range)
    
    tf_idf_features(X_train_,X_test_ , target, ngram_range)

    corpus_train = preprocess_documents(list(X_train_))
    corpus_test = preprocess_documents(list(X_test_))
    
    doc_embedding_featues(corpus_train, corpus_test, n_workers, target, ngram_range)
    
    word_embedding_features(corpus_train, corpus_test, X_train_, X_test_, n_workers, target, ngram_range)
        

#==================================================================================================
# Execute features generation 
# 

def set_feature_types(lda_ntopics=100, run_lda = True, run_general = True):
    """
    Generates all five blocks of features for each target variable and n-gram range. Stores in `features/` folder in pickle format.
    **NOTE: Execution of the next block may take several hours**
    """
    targets = ({'target': 'control', 'fname': 'train-contcoll.csv'},
            {'target': 'collaboration', 'fname': 'train-contcoll.csv'})
    ranges = ((1,2), (1,3))
    params = ()

    for t in targets:
        for r in ranges:
            t['ngram_range'] = r
            params = params + (t.copy(),)


    for p in params:
        if run_lda:
            lda_features(target=p['target'],
                        fname=p['fname'],
                        ngram_range=p['ngram_range'],
                        lda_ntopics=lda_ntopics , 
                        subset = 0.25)
        if run_general:
            general_features(target=p['target'],
                                fname=p['fname'],
                                ngram_range=p['ngram_range'],
                                subset = 0.25) 


def plot_save_perplexities(lda_ntopics_list, perplexities, p, save_dir='lda_search'):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    fname_end = "lda_ngram_%02d" % max(p['ngram_range'])
    fname_front=save_dir + os.path.sep + p['target']
    plt.figure()
    plt.plot(lda_ntopics_list, perplexities, color='purple', marker='.', linestyle='dashed')
    plt.xlabel('Number of Topics')
    plt.ylabel('Perplexity')
    plt.title("Perplexity " + p['target'] + " ngram "+str(max(p['ngram_range'])))
    plt.savefig("%s_perplexity_%s.png" % (fname_front,fname_end ), dpi=600)
    plt.close()           
    rpc = [None]
    top_prev, perp_prev = lda_ntopics_list[0], perplexities[0]
    for i in range(len(lda_ntopics_list)):
        if i>0:
            rpc.append((perplexities[i] - perp_prev)/(lda_ntopics_list[i] - top_prev))
            perp_prev = perplexities[i]
            top_prev = lda_ntopics_list[i]
    plt.figure()
    plt.plot(lda_ntopics_list[1:], rpc[1:], 'p--')
    plt.xlabel('Number of Topics')
    plt.ylabel('RPC')
    plt.title("RPC " + p['target'] + " ngram "+str(max(p['ngram_range'])))
    plt.savefig("%s_RPC_%s.png" % (fname_front,fname_end ), dpi=600)
    plt.close()           
    df = pd.DataFrame(columns =['N LDA Topics', 'Perplexity', 'RPC'], data =zip(lda_ntopics_list, perplexities, rpc))
    with open("%s_topic_search_ %s.csv" % (fname_front,fname_end ), 'w') as file:
        df.to_csv(file, index=False)

def search_lda_topic_n():
    targets = ({'target': 'control', 'fname': 'train-contcoll.csv'},)
    ranges = ((1,2),)
    params = ()

    for t in targets:
        for r in ranges:
            t['ngram_range'] = r
            params = params + (t.copy(),)

    lda_ntopics_list = [3,7, 10, 15, 20, 25, 30]         # Number of LDA topics to try

    topic_perplexity = [] # list of tuple of (number of topics, perplexity)

    # for each dataset for each set of ngram ranges, find the best topic number   
    perplexities = [] 
    for p in params:
        for lda_ntopics in lda_ntopics_list:
            perplexity = lda_features(target=p['target'],
                            fname=p['fname'],
                            ngram_range=p['ngram_range'],
                            lda_ntopics=lda_ntopics , 
                            subset = 0.25)
            perplexities.append( perplexity)
        plot_save_perplexities(lda_ntopics_list, perplexities, p)


if __name__=="__main__":
    search_lda_topic_n()
    #set_feature_types()    