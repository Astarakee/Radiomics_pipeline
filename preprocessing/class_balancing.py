from imblearn.over_sampling import SMOTE




def smote_balancing(feature_set, label_set):
    '''
    balancing the imabalanced dataset by synthesizing new data to the 
    minority class labels.

    Parameters
    ----------
    feature_set : array
        feature matrix of all data.
    label_set : array
        label vector representing all data.

    Returns
    -------
    feature_set_balanced : array
        augmented feature set.
    label_set_balanced : array
        augmented label vector.

    '''
    
    sm = SMOTE(sampling_strategy='auto',  random_state=None)
    feature_set_balanced, label_set_balanced = sm.fit_sample(feature_set, label_set)  
    
    return feature_set_balanced, label_set_balanced
