import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from ReliefF import ReliefF
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_classif
from mlxtend.feature_selection import SequentialFeatureSelector as SFS




def sequential_selection(clf_model, n_features=3, foward_state=True, floating_state=False,
                         metric='roc_auc', k_fold=3, n_jobs=-1):
    '''
    integrating sequential feature selection algorithms into a learning algorithm
    Depending on the size of the feature matrix and CPU resources,
    it would take from minutes to days!

    Parameters
    ----------
    clf_model : class
        a compiled learning algorithm with sklearn format.
        can be selected from the models in learning algorithm module.
    n_features : int
        number of features to be selected.
    foward_state : bool
        set True for fwd and False for bwd selection.
    floating_state : bool
        set True for floating-based selection.
    metric : str
        can be selected from 'accuracy', 'f1', 'precision', 'recall', 'roc_auc'.
    k_fold : int
        number of folds in k-fold cross validation.
    n_jobs : int
        number of parallel jobs.
        Use -1 for computational time. In general, the higher the number
        of employed threades, the less time needs for computation.

    Returns
    -------
    sfs_model : class
        A compiled SFS model with the defined settings.
        It should be fitted on the training dataset afterwards,

    '''
    
    
    sfs_model = SFS(clf_model,
                     k_features = n_features,
                     forward = foward_state,
                     floating = floating_state,
                     scoring = metric,
                     n_jobs = n_jobs,
                     cv= k_fold)
    return sfs_model



def nonconstants(x_train, x_val, x_test = None, thr=0.8):
    '''
    Removing Constant Features using Variance Threshold
    
    Parameters
    ----------
    thr : float
        thr is calculated based on variance each feature. Those features with a
        variance lower than this thr considered as constatnt and will be removed.
    x_train : array
        feature matrix of training set.
    x_val : array
        feature matrix of validation set.
    x_test : array, optional
        feature matrix of testing set. The default is None.

    Returns
    -------
    non_constant : dict
        updated feature sets after removing constant features.

    '''

    non_constant = {}
    constant_filter = VarianceThreshold(threshold = thr)
    constant_filter.fit(x_train)
    #nonconstant_index = constant_filter.get_support(indices = True)
    nonconstant_train = constant_filter.transform(x_train)
    nonconstant_val = constant_filter.transform(x_val)
    non_constant['train'] = nonconstant_train
    non_constant['val'] = nonconstant_val
    if x_test is not None:
        nonconstant_test = constant_filter.transform(x_test)
        non_constant['test'] = nonconstant_test
    
    return non_constant





def uncorrelateds(x_train, x_val, x_test=None, corr_thresh=0.8):
    '''
    Calculate the correlation matrix between the features
    and remove those with correlation higher than thresholds.
    e.g., corr_thresh = 0.95
    
    Parameters
    ----------
    corr_thresh : float
        threshold based on correlation values.
    x_train : array
        feature matrix of training set..
    x_val : array
        feature matrix of validation set..
    x_test : array, optional
        feature matrix of test set.. The default is None.

    Returns
    -------
    uncorrelated : dict
        updated feature sets after removing highly correlated features.

    '''
    uncorrelated = {}
    correlated_ind = []
    uncorrelated_ind = []
    df_x_train = pd.DataFrame(x_train)
    correlation_matrix = df_x_train.corr()

    for ind in range(correlation_matrix.shape[1]):
        for ix in range(ind):
            if abs(correlation_matrix.iloc[ind, ix]) > corr_thresh:
                colname = correlation_matrix.columns[ind]
                correlated_ind.append(colname)
            else:
                column_name = correlation_matrix.columns[ind]
                uncorrelated_ind.append(column_name)
                    
                
    correlated_ind = list(set(correlated_ind))
    uncorr_x_train = np.delete(x_train, correlated_ind, axis = 1)
    uncorr_x_val = np.delete(x_val, correlated_ind, axis = 1)
    uncorrelated['train'] = uncorr_x_train
    uncorrelated['val'] = uncorr_x_val
    
    if x_test is not None:
        uncorr_x_test = np.delete(x_test, correlated_ind, axis = 1)
        uncorrelated['test'] = uncorr_x_test
    uncorrelated['feature_indices'] = uncorrelated_ind.tolist()
    return uncorrelated



def lasso(x_train, y_train, x_val, x_test=None, n_fold=5, max_iters=50, thr=0.5):
    '''
    Least Absolute Shrinkage and Selection Operator (LASSO) 
    Parameters
    ----------
    n_fold : int
        the best model comes from cross validation on training set.
    max_iters : int
        The maximum number of iterations, e.g., 2000
    thr : float
        features with importance greater or equal are kept. e.g., 0.6
    x_train : array
        feature matrix of training set.
    y_train : array
        vector representing class labels of training set.
    x_val : array
        feature matrix of validation set.
    x_test : array, optional
        feature matrix of validation set. The default is None.

    Returns
    -------
    lasso : dict
        updated feature sets after removing sparse coefficients

    '''
    lasso = {}
    cv_lasso = LassoCV(cv = n_fold, max_iter = max_iters, n_jobs = 1)
    cv_lasso_model = SelectFromModel(cv_lasso, threshold = thr)
    cv_lasso_model.fit(x_train, y_train)
    #n_remained_lasso = cv_lasso_model.transform(x_train).shape[1]
    remained_lasso_idx = cv_lasso_model.get_support(indices = True)
    x_train_lasso = x_train[:,remained_lasso_idx] 
    x_val_lasso = x_val[:,remained_lasso_idx]
    lasso['train'] = x_train_lasso
    lasso['val'] = x_val_lasso
    
    if x_test is not None:
        x_test_lasso = x_test[:,remained_lasso_idx]
        lasso['test'] = x_test_lasso
    lasso['feature_indices'] = remained_lasso_idx.tolist()
    return lasso



def relief( x_train, y_train, x_val, x_test = None, n_neighbors=5, n_features=10):
    '''
    Extended Relevance in Estimatory Features (RELIEF)

    Parameters
    ----------
    n_neighbors : int
        k-nearest neighbors.
    n_features : int
        number of features to be kept.
    x_train : array
        feature matrix of training set.
    y_train : array
        Vector representng class labels of training set.
    x_val : array
        feature matrix of validation set.
    x_test : array, optional
        feature matrix of testing set. The default is None.

    Returns
    -------
    relief_features : dict
        updated feature sets after removing some features.

    '''
    relief_features = {}
    relief = ReliefF(n_neighbors = n_neighbors, n_features_to_keep = n_features)
    relief.fit(x_train, y_train)
    relief_train = relief.transform(x_train)
    relief_val = relief.transform(x_val)
    relief_features['train'] = relief_train
    relief_features['val'] = relief_val
    if x_test is not None:
        relief_test = relief.transform(x_test)
        relief_features['test'] = relief_test
    
    return relief_features


def pca_linear(x_train, x_val, x_test = None, n_component=10):
    '''
    Dimensionality redocution with linear PCA

    Parameters
    ----------
    n_component : int
        number of tansformed features to be kept.
    x_train : array
        feature matrix of training set.
    x_val : array
        feature matrix of validation set.
    x_test : array, optional
        feature matrix of test set. The default is None.

    Returns
    -------
    pca_features : dict
        transformed features into new space with lower dimension.

    '''
    
    pca_features = {}
    pca = PCA(n_components = n_component)
    pca.fit(x_train)
    x_train_pca = pca.transform(x_train)
    x_val_pca = pca.transform(x_val)
    pca_features['train'] = x_train_pca
    pca_features['val'] = x_val_pca
    if x_test is not None:
        x_test_pca = pca.transform(x_test)
        pca_features['test'] = x_test_pca
    
    return pca_features


def pca_kernel(x_train, x_val, x_test = None, n_component=10, kernel_name='rbf'):
    '''
    Dimensionality redocution with kernel-based PCA

    Parameters
    ----------
    n_component : int
        number of tansformed features to be kept..
    kernel_name : str
        should be selected from ‘poly’, ‘rbf’, ‘sigmoid’, ‘cosine’,
        or even 'linear'!
    x_train : array
        feature matrix of training set.
    x_val : array
        feature matrix of validation set.
    x_test : array, optional
        feature matrix of testing set. The default is None.

    Returns
    -------
    pca_features : dict
        transformed features into new space with lower dimension.

    '''
    
    pca_features = {}
    pca = KernelPCA(n_components = n_component, kernel = kernel_name)
    pca.fit(x_train)
    x_train_pca = pca.transform(x_train)
    x_val_pca = pca.transform(x_val)
    pca_features['train'] = x_train_pca
    pca_features['val'] = x_val_pca
    if x_test is not None:
        x_test_pca = pca.transform(x_test)
        pca_features['test'] = x_test_pca
        
    return pca_features

    

def mutual_info(x_train, y_train, x_val, x_test = None, n_neighbor=5):
    '''
    Quantifying nonlinear dependency between the features and labels in training
    set and keep those features which are independent (MI_coef=0)

    Parameters
    ----------
    n_neighbor : int
        number of neighbors to use for MI estmiation of continuous variables.
    x_train : array
        feature matrix of training set.
    y_train : array
        label vector representing class labels of training set.
    x_val : array
        feature matrix of validation set.
    x_test : array, optional
        feature matrix of test set. The default is None.

    Returns
    -------
    mi_features : dict
        updated feature sets after removing dependent features.

    '''
    
    mi_features = {}
    mi_model = mutual_info_classif(x_train, y_train,
                                   n_neighbors = n_neighbor,
                                   copy = True)
    indpndts = np.where(mi_model==0)[0]
    mi_x_train = np.take(x_train, indpndts, axis=1)
    mi_x_val = np.take(x_val, indpndts, axis=1)
    mi_features['train'] = mi_x_train
    mi_features['val'] = mi_x_val
    if x_test is not None:
        mi_x_test = np.take(x_test, indpndts, axis=1)
        mi_features['test'] = mi_x_test
    mi_features['feature_indices'] =  indpndts.tolist()
    return mi_features



        

