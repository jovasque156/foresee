from sklearn import metrics
import pandas as pd
import numpy as np

#Defining the Grup Fair metrics
def accuracy(y_true, y_pred):
    return metrics.accuracy_score(y_true, y_pred)

def auc(y_true, y_pred):
    return metrics.roc_auc_score(y_true, y_pred)

def f1score(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred)

def recall(y_true, y_pred):
    #it returns the recall/TPR
    #it is assumed that positive class is equal to 1
    
    return y_pred[y_true==1].sum()/y_true.sum()

def fpr(y_true, y_pred):
    #It returns the False Positive Rate
    #it is assumed that positive class is equal to 1
    
    return y_pred[y_true==0].sum()/(len(y_true)-y_true.sum())

def precision(y_true, y_pred):
    #It returns the precision
    
    return y_true[y_pred==1].sum()/y_pred.sum()

def selection_rate(y_pred, protected_attr, priv_class, unpriv_class=None):
    #It returns the Selection Ratio for priviliged and unpriviliged group
    #The positive class must be equal to 1, which is used for 'select' the individual
    #Pr(h=1|priv_class=a)
    
    overall = y_pred.sum()/len(y_pred)
    if unpriv_class == None:
        unpr=y_pred[~(protected_attr==priv_class)].sum()/len(y_pred[~(protected_attr==priv_class)])
    else:
        unpr=y_pred[protected_attr==unpriv_class].sum()/len(y_pred[protected_attr==unpriv_class])
        
    priv=y_pred[protected_attr==priv_class].sum()/len(y_pred[protected_attr==priv_class])
    
    return overall, unpr, priv

def demographic_parity_dif(y_pred, protected_attr, priv_class, unpriv_class=None):
    #It returns the Statistical Parity Difference considering the prediction
    #It is assumed that positive class is equal to 1
    #Pr(h=1|priv_class=unpriviliged) - Pr(h=1|priv_class=priviliged)
    
    _, unpr, priv = selection_rate(y_pred, protected_attr, priv_class, unpriv_class)
    
    return unpr-priv

def disparate_impact_rate(y_pred, protected_attr, priv_class, unpriv_class=None):
    #It returns the Disparate Impact Ratio
    #It is assumed that positive class is equal to 1
    # Pr(h=1|priv_class=unpriviliged)/Pr(h=1|priv_class=priviliged)
    # Note that when Disparate Impact Ratio<1, it is considered a negative impact to unpriviliged class
    # This ratio can be compared to a threshold t (most of the time 0.8 or 1.2) in order to identify the presence
    # of disparate treatment.
    
    _, unpr, priv = selection_rate(y_pred, protected_attr, priv_class, unpriv_class)
    
    return unpr/priv


def equal_opp_dif(y_true, y_pred, protected_attr, priv_class, unpriv_class=None, weight=False):
    #It returns the Equal Opportunity Difference between the priv and unpriv group
    #This is obtained by substracting the recall/TPR of the priv group to the recall/TPR of the unpriv group
    
    
    tpr_priv = recall(y_true[protected_attr==priv_class], y_pred[protected_attr==priv_class])
    if unpriv_class == None:
        tpr_unpriv = recall(y_true[protected_attr!=priv_class], y_pred[protected_attr!=priv_class])
    else:
        tpr_unpriv = recall(y_true[protected_attr==unpriv_class], y_pred[protected_attr==unpriv_class])
    
    return tpr_unpriv-tpr_priv 

def equalized_odd_dif(y_true, y_pred, protected_attr, priv_class, unpriv_class = None):
    pos_priv = y_pred[(y_true == 1) & (protected_attr == priv_class)].mean()
    pos_unpr = y_pred[(y_true == 1) & (protected_attr != priv_class)].mean()
    neg_priv = y_pred[(y_true == 0) & (protected_attr == priv_class)].mean()
    neg_unpr = y_pred[(y_true == 0) & (protected_attr != priv_class)].mean()
    
    #if np.isnan(pos_priv): pos_priv=0
    #if np.isnan(pos_unpr): pos_unpr=0
    #if np.isnan(neg_priv): neg_priv=0
    #if np.isnan(neg_unpr): neg_unpr=0
    
    
    pos = np.abs(pos_priv - pos_unpr)
    neg = np.abs(neg_priv - neg_unpr)
    
    return (pos + neg)*0.5
    
def sufficiency_dif(y_true, y_pred, protected_attr, priv_class, unpriv_class = None):
    #It returns the Sufficiency difference between priv and unpriv groups
    #It is assumed that the positive class is equal to 1
    # Pr(Y=1|h=1,priv_class=unpriv) - Pr(Y=1|h=1,priv_class=priv)
    
    prec_priv = precision(y_true[protected_attr==priv_class], y_pred[protected_attr==priv_class])
    if unpriv_class == None:
        prec_unpr = precision(y_true[protected_attr!=priv_class], y_pred[protected_attr!=priv_class])
    else:
        prec_unpr = precision(y_true[protected_attr==unpriv_class], y_pred[protected_attr==unpriv_class])
    
    return prec_unpr-prec_priv
    
def discrimanation(y_true, y_pred, protected_attr, priv_class, unpriv_class = None):
    '''
    Here discrimination is understood as the level of misclassification
    '''
    misclassification_unpriv = 1
    misclassification_priv = 1
    if sum(1*(protected_attr!=priv_class))!=0:
        misclassification_unpriv = (y_true[protected_attr!=priv_class]!=y_pred[protected_attr!=priv_class]).sum()/len(y_true[protected_attr!=priv_class])
            
    if sum(1*(protected_attr==priv_class))!=0:
        misclassification_priv=(y_true[protected_attr==priv_class]!=y_pred[protected_attr==priv_class]).sum()/len(y_true[protected_attr==priv_class])
    
    return misclassification_unpriv, misclassification_priv