#Data Handling
import pandas as pd
import numpy as np

#Pipelines
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack
from scipy import sparse
from scipy.sparse import csr_matrix
import scipy
from sklearn.model_selection import GridSearchCV

#Transformation
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

#Imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer

#Decomposition
from sklearn.decomposition import TruncatedSVD

#Storing estimators
import pickle

#Feature Selection
import statsmodels.api as sm

def rm_outliers(X, num_sd=2):
    mean, ds = np.mean(X), np.std(X)
    lowr, upper = mean - ds* num_sd, mean + ds*num_sd
    r = [x for x in X if x <= upper and x >= lower]
    return r

def deletCorrVar(x, del_col_per_it):
    #del_col_per_it contains the variable to delete for each iterations.
    #the function returns the x with after delete variables in del_col_per_it.
    
    x=pd.DataFrame(x.toarray()).values

    for i in del_col_per_it:
        x = np.delete(x,i,1)
    return x

def backwardElimination(x, Y, sl):
    #Receives dataset x, dependent variable Y, and the pvalue threshold of sl
    #Returns the set of variables which does not allow to "recreate" the variable

    x = pd.DataFrame(x.toarray()).values
    Y = pd.DataFrame(Y).values

    del_col_per_it = []
    min_pvals = []
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        minVar = min(regressor_OLS.pvalues).astype(float)
        if minVar <= sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == minVar):
                    x = np.delete(x, j, 1)
                    del_col_per_it.append(j)
                    min_pvals.append((j,minVar))



    regressor_OLS.summary()
    return x, del_col_per_it, min_pvals

def applypreprocessing(X, nompipe, numpipe, idnumerical = None, idnominal = None):
    if idnumerical==None:
        idnumerical = [i for i, e in enumerate(X.dtypes) if e == 'float64']
    if idnominal==None:
        idnominal = list(np.setdiff1d(list(range(0,len(X.columns))),idnumerical))
    
    numerical = list(X.iloc[:,idnumerical].columns)
    nominal = list(X.iloc[:,idnominal].columns)
    
    #Identifying numerical and nominal variables
    X_nom = X.loc[:,nominal]
    
    #Numerical
    X_num = X.loc[:,numerical]
    
    #Apply trained pipes
    if nompipe==None:
        X_num = numpipe.transform(X_num)
        X_final = X_num
    elif numpipe==None:
        X_nom = nompipe.transform(X_nom)
        if not(scipy.sparse.issparse(X_nom)):
            X_nom = sparse.csr_matrix(X_nom)
        X_final = X_nom
    else:
        X_nom = nompipe.transform(X_nom)
        if not(scipy.sparse.issparse(X_nom)):
            X_nom = sparse.csr_matrix(X_nom)
        X_num = numpipe.transform(X_num)
        X_final = hstack((X_num, X_nom))   
    
    return X_final

def preprocessing(X, idnumerical=None, idnominal=None, imputation=True, encode = 'one-hot', normalization = True ):
    #Return a sparse matrix using X as a train dataset for fitting estimators.
    #Additionally it is returned the fitted pipelines related to numerical and nominal features.
    
    #Identifying numerical and nominal variables
    if idnumerical==None:
        idnumerical = [i for i, e in enumerate(X.dtypes) if e == 'float64']
    if idnominal==None:
        idnominal = list(np.setdiff1d(list(range(0,len(X.columns))),idnumerical))
        
    numerical = list(X.iloc[:,idnumerical].columns)
    nominal = list(X.iloc[:,idnominal].columns)
    nom_num = [y for x in [numerical, nominal] for y in x] 
    
    X_nom = X.loc[:,nominal]
    X_num = X.loc[:,numerical]
    

    #Applying estimators for nominal an numerical
    #nominal
    pipe_nom = None
    if len(nominal)>0:
        estimators = []
        imp_nom = SimpleImputer(strategy='most_frequent')
        scale = StandardScaler(with_mean=True)
        if encode == 'one-hot':
            enc = OneHotEncoder(drop='first')
        elif encode == 'label':
            enc = OrdinalEncoder()

        if imputation == True:
            estimators.append(('imputation', imp_nom))
        if encode != None:
            estimators.append(('encoding', enc))
        if normalization:
            estimators.append(('standardscale', scale))

        pipe_nom = Pipeline(estimators)
        pipe_nom.fit(X_nom)
    
    #numerical
    pipe_num = None
    if len(numerical)>0:
        estimators = []
        imp_num = IterativeImputer(max_iter=100, random_state=1)
        scale = StandardScaler(with_mean=True)

        if imputation == True:
            estimators.append(('impuation', imp_num))

        if normalization:
            estimators.append(('standardscale', scale))

        pipe_num = Pipeline(estimators)
        pipe_num.fit(X_num)
        
    #Merge both transformations
    if len(nominal)<1:
        X_num = pipe_num.transform(X_num)
        X_final = X_num
    elif len(numerical)<1:
        X_nom = pipe_nom.transform(X_nom)
        if not(scipy.sparse.issparse(X_nom)):
            X_nom = sparse.csr_matrix(X_nom)
        X_final = X_nom
    else:
        X_nom = pipe_nom.transform(X_nom)
        if not(scipy.sparse.issparse(X_nom)):
            X_nom = sparse.csr_matrix(X_nom)
        X_num = pipe_num.transform(X_num)
        X_final = hstack((X_num, X_nom))    
    
    hot_encoder = pipe_nom['encoding']
    if encode == 'one-hot':
        nom_features = hot_encoder.get_feature_names(nominal)
    else:
        nom_features = nominal
        
    return X_final, pipe_nom, pipe_num, numerical, nom_features


def svd_decomposition(X, thres = 0.95):
    #Return a sparse matrix using X as a train dataset for fitting estimators.
    #Additionally it is returned the fitted pipelines related the decomposition of X.
    
    svd_dec = TruncatedSVD(n_components=X.shape[1]-1, n_iter=100, random_state=1) 
    svd_dec.fit(X)

    #Determining the number of components
    n_comp = 0
    tota_var = 0
    i=0
    while tota_var<thres:
        n_comp += 1
        tota_var += svd_dec.explained_variance_ratio_[i]
        i+=1

    svd_dec = TruncatedSVD(n_components=n_comp, n_iter=100, random_state=1) 
    X_dec = svd_dec.fit_transform(X)
    
    return X_dec, svd_dec

def applysvd_decomposition(X, svd_estimator):
    return svd_estimator.transform(X)

def import_pickle(directory):
    with open(directory, 'rb') as f:
        p = pickle.load(f)

    return p

def get_grid(X, y, parameters, model, model_name, scoring=['f1'], refit = 'f1'):
    pipe_model_train = Pipeline([(model_name, model)])
    
    grid = GridSearchCV(pipe_model_train, param_grid=parameters, cv=5, scoring = scoring, refit=refit) 
    #alterantives: 'accuracy', 'roc_auc' and 'f1' (the las two only accept binary variables)
    
    fit = grid.fit(X,y)
    
    return fit

def center(X):
    xmean = X.mean(axis=0)
    xmean = xmean[None, ...]

    #self.xmean = xmean

    return X - xmean

def normalize(X):
    x2sum = (X ** 2).sum(0)
    x2sum = x2sum[None, ...]

    #self.x2sum = x2sum

    return X / np.sqrt(x2sum)