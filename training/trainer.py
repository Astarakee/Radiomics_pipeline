import sys
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.append('../../radiomics_pipeline/utilities/')
from metircs_evaluation import conf_matrix, metrics


def learning(clf, x_train, y_train, x_val, y_val, x_test=None):
    '''

    Parameters
    ----------
    clf : class
        A compiled learning algorithms based on sklearn format.
    x_train : array
        A matrix of the training features.
    y_train : array
        A vector representing the class labels of the training set.
    x_val : array
        A matrix of the validation features.
    y_val : array
        A vector representing the class labels of the validation set.
    x_test : array, optional
        A matrix of the validation features. The default is None.

    Returns
    -------
    summary : dict
        metrics as well as predicted values of validation
        and test (optional) sets.

    '''
    
    summary = {}
    
    clf.fit(x_train, y_train)
    y_val_pred = clf.predict(x_val)
    y_val_pred_prob = clf.predict_proba(x_val)[:,1]
    
    try:
        roc_auc_val = roc_auc_score(y_val, y_val_pred_prob)
    except ValueError:
        roc_auc_val = -999
    y_val_pred_prob = y_val_pred_prob.tolist()
    confusion_matrix = conf_matrix(y_val, y_val_pred)
    accuracy, sensitivity, specificity = metrics(confusion_matrix)
    
    if x_test is not None:        
        y_test_pred_prob = clf.predict_proba(x_test)[:,1]
        y_test_pred_prob = y_test_pred_prob.tolist()
    else:
        y_test_pred_prob = None

    
    summary['val_accuracy'] = accuracy
    summary['val_sensitivity'] = sensitivity
    summary['val_specificity'] = specificity
    summary['val_auc'] = roc_auc_val
    summary['val_pred_prob'] = y_val_pred_prob
    summary['test_pred_prob'] = y_test_pred_prob
    
    return summary
    
    

def learning_with_sfs(sfs_model, feature_set, label_set, n_features=3):
    '''
    

    Parameters
    ----------
    sfs_model : class
        A compiled SFS model based on a sklearn algorithm.
    feature_set : array
        feature matrix of all data.
    label_set : array
        label vector representing the class labels.
    n_features : int
        number of features to be selected.

    Returns
    -------
    summary : dict
        returns a summary of the SFS model including the selected features,
        and performance of the model with the selected features.

    '''
    
    summary = {}
    sfs_fit = sfs_model.fit(feature_set, label_set)
    sfs_features = sfs_fit.subsets_  
    sfs_features = sfs_features[n_features]
    cv_scores = sfs_features['cv_scores']
    cv_average_score = np.mean(cv_scores)
    cv_std_score = np.std(cv_scores)
    
    selected_features = sfs_features['feature_idx']
    exp_name = str(n_features)+' selected features'
    metric_name_mean = exp_name+'_score_cv_mean'
    metric_name_std = exp_name+'_score_cv_std'
    metric_name_features = exp_name+'_names'
    summary[metric_name_mean] = cv_average_score
    summary[metric_name_std] = cv_std_score
    summary[metric_name_features] = selected_features
    
    return summary
    
