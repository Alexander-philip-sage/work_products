#**************************************************************************************************
# BOARD LEADERSHIP VARIABLE
# FILE: MODEL TRAINING
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
from sklearn.metrics import mean_squared_error

from gensim.parsing.preprocessing import preprocess_documents
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec

from features import  search_lda_topic_n, set_feature_types

###################################################################################################
# GLOBAL PARAMETERS
###################################################################################################

cv_folds=5              # number of folds for cross-validation
cv_test=0.20            # fraction of test data in cross-validation
score='accuracy'        # Main metric for modeling
max_iter=1e3            # maximum number of iterations allowed (CV and TF-IDF)
from features import random_state


###################################################################################################
# STEP 3 (MODELING)
###################################################################################################

# Utility functions to plot ROC curve and calculate G-means & optimal thresholds:

def plot_roc(estimator, X_test, y_test):
    
    """
    Creates ROC curve for estimator.
    
    `X_test`: test covariates
    `y_test`: test target
    
    Returns plot for ROC curve.
    """
    
    probs = estimator.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)     ### THIS IS WHERE THE AUC IS BEING COMPUTED
    
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    
    """
    Calculates G-mean and optimal threshold
    """

    gmeans = np.sqrt(tpr * (1-fpr))
    
    return plt,gmeans,threshold

#==================================================================================================
# Configuration

def ml_config():
    """
    Generates all combinations of target variables, n-gram ranges, and features.

    Example Below
    config = pd.DataFrame({'model': [['lg', 'rf']],
                        'ngram': [[2, 3, 5, 10]],
                        'lda_ntopics': 100,
                        'features': [[('cv',), 
                                        ('tfidf',), 
                                        ('word2vec',),
                                        ('doc2vec',),
                                        ('lda',),
                                        ('cv','tfidf','word2vec',),
                                        ('cv','tfidf','doc2vec',),
                                        ('cv','tfidf','lda',),
                                        ('cv','tfidf','word2vec','lda',),
                                        ('cv','tfidf','doc2vec','lda',)]]})
    """

    config = pd.DataFrame({'model': [['lg', 'rf']],
                        'ngram': [[2, 3]],
                        'lda_ntopics': [[100,]],
                        'features': [[('word2vec',),
                                        ('doc2vec',),
                                        ('lda',)]]})

    config = config.explode('model')                .explode('ngram')                .explode('features').drop_duplicates().reset_index(drop=True)
    config['target'] = np.tile(['control', 'collaboration'], (len(config),1)).tolist()
    config = config.explode('target').drop_duplicates().reset_index(drop=True)
    return config
#==================================================================================================
# Model Training 
#
# Executes model training for created config. All results which includes ROC curves plots, classification reports, and confusion matrices are saved to `results` folder.
# 
# Main outcome is `output/results.csv` file.
# 
# **NOTE: Execution of next block may take several hours**


def train_models(config):
    if not os.path.exists('output'):
        os.mkdir('output')
    print
    for _, row in config.iterrows():
        
        # Defines filename
        row['model_features'] = '_'.join(row['features'])    
        fname = 'output/%s-%s-%s-ngram%02d' % (row['target'], 
                                            row['model'], 
                                            row['model_features'],
                                            row['ngram'])
        
        # Checks if we should generate this model
        if os.path.exists('%s_model.pickle' % fname):
            print('Skipping "%s"...' % fname)
            continue
        else:
            print('Generating "%s"...' % fname)
        
        # Reads target values
        with open('features/%s_train_target_ngram_%02d.pickle' % (row['target'], row['ngram']), 'rb') as f:
            y_train = pickle.load(f)
        with open('features/%s_test_target_ngram_%02d.pickle' % (row['target'], row['ngram']), 'rb') as f:
            y_test = pickle.load(f)
            
        X_train = None
        X_test = None
        print("config")
        print(config['lda_ntopics'][0])
        print(type(config['lda_ntopics'][0]))
        for feature in row['features']:

            if feature=='lda':
                with open('features/%s_train_%s_ngram_%02d_ntop_%03d.pickle' % (row['target'], feature, row['ngram'], config['lda_ntopics'][0]), 'rb') as f:
                    X_train = hstack([X_train, pickle.load(f)])
                with open('features/%s_test_%s_ngram_%02d_ntop_%03d.pickle' % (row['target'], feature, row['ngram'], config['lda_ntopics'][0]), 'rb') as f:
                    X_test = hstack([X_test, pickle.load(f)])
            else:
                with open('features/%s_train_%s_ngram_%02d.pickle' % (row['target'], feature, row['ngram']), 'rb') as f:
                    X_train = hstack([X_train, pickle.load(f)])
                with open('features/%s_test_%s_ngram_%02d.pickle' % (row['target'], feature, row['ngram']), 'rb') as f:
                    X_test = hstack([X_test, pickle.load(f)])

        # Defines estimator
        if row['model']=='lg':
            estimator = LogisticRegression(random_state=random_state, 
                                        max_iter=max_iter, 
                                        multi_class='ovr',
                                        class_weight='balanced')
            tuned_parameters = [{'solver': ['lbfgs'],
                                'penalty': ['l2'], 
                                'class_weight': ['balanced', None],
                                'C': [0.001, 0.01, 0.1, 1]},
                                {'solver': ['liblinear'],
                                'penalty': ['l1'], 
                                'class_weight': ['balanced', None],
                                'C': [0.001, 0.01, 0.1, 1]}]
        else:
            estimator = RandomForestClassifier(random_state=random_state,
                                            n_jobs=os.cpu_count())
            tuned_parameters = [{'bootstrap': [True, False],
                                'max_depth': [10, 50, None],
                                'n_estimators': [100, 500, 1000]}]
        
        # Cross-validation setup
        cv = ShuffleSplit(n_splits=cv_folds, test_size=cv_test, random_state=random_state)
        
        # Hyper-parameters tuning
        clf = GridSearchCV(estimator, tuned_parameters, scoring=score, cv=cv)
        best_fit = clf.fit(X_train, y_train)

        # Scoring
        y_pred = clf.predict(X_test)
        
        # Updates results
        row['accuracy_train'] = best_fit.score(X_train, y_train)
        row['accuracy_test'] = best_fit.score(X_test, y_test)
        row['loss_test'] = mean_squared_error(y_test,y_pred)

        # Saves report & plot
        with open('%s.txt' % fname, 'w') as f:
            f.write('# Best parameters:\n')
            json.dump(clf.best_params_, f)
            f.write('\n\n# Classification Report:\n')
            f.writelines(metrics.classification_report(y_test, y_pred, zero_division=0))
            f.write('\n\n# Confusion Matrix:\n')
            f.writelines(pd.DataFrame(metrics.confusion_matrix(y_test, y_pred)).to_csv(sep='\t'))
        
        plt,gmeans,thresholds = plot_roc(clf, X_test, y_test)
        plt.savefig('%s.png' % fname, dpi=600)
        plt.close()
        
        
        ix = np.argmax(gmeans)
        print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
        row['G-Mean'] = gmeans[ix]
        row['Threshold'] = thresholds[ix]

        first_record = os.path.exists('output/results.csv')
        pd.DataFrame(row[['target','model','model_features','ngram','accuracy_train','accuracy_test','loss_test','G-Mean',"Threshold"]])     .T.to_csv('output/results.csv',
                index=False,
                header=True if first_record==0 else False,
                mode='w' if first_record==0 else 'a')
        
        # Saves model
        with open('%s_model.pickle' % fname, 'wb') as f:
            pickle.dump(best_fit, f)
            
        # Saves wrong predictions
        with open('features/%s_test_X_ngram_%02d.pickle' % (row['target'], row['ngram']), 'rb') as f:
            X_test_ = pickle.load(f).reset_index(drop=True)
            
        wrong_predicts = pd.DataFrame(np.column_stack((best_fit.predict(X_test), 
                                                    best_fit.predict_proba(X_test), 
                                                    y_test)),
                                    columns=['predicted_class', 'prob0', 'prob1', 'true_class'])
        wrong_predicts = pd.concat([wrong_predicts, X_test_], axis=1)
        wrong_predicts = wrong_predicts[wrong_predicts['predicted_class'] != wrong_predicts['true_class']]                      .sort_values(['true_class']).reset_index(drop=True)
        wrong_predicts.to_csv('%s_wrong_predicts.csv' % fname, index=False)

if __name__=="__main__":

    
    set_feature_types()
    config = ml_config()
    train_models(config)
    # END
