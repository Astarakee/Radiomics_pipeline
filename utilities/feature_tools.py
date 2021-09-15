import numpy as np
from sklearn import preprocessing


def train_val_split(feature_set, label_set, train_index, test_index):
    '''
    Splitting the data into train/val sets for cross validation

    Parameters
    ----------
    feature_set : array
        feature matrix containing all data.
    label_set : array
        vector of labels representing all data.
    train_index : list
        list of training indices generated by sklearn "KFold.split".
    test_index : list
        list of validation indices generated by sklearn "KFold.split"..

    Returns
    -------
    x_train : array
        subset of feature set as training set.
    x_val : array
        subset of feature set as validation set.
    y_train : TYPE
        subset of label set as training labels.
    y_val : array
        subset of label set as validation labels.

    '''
    
    x_train = np.asarray(list(feature_set[i] for i in train_index))  
    x_val = np.asarray(list(feature_set[i] for i in test_index))
    y_train = np.asarray(list(label_set[i] for i in train_index))
    y_val = np.asarray(list(label_set[i] for i in test_index))
    
    return x_train, x_val, y_train, y_val


def feature_normalization(x_train, x_val = None, x_test = None):
    '''
    Normalize the each feature into the range of 0 to 1

    Parameters
    ----------
    x_train : array
        Feature matrix of training set.
    x_val : array
        Feature matrix of validation set.
    x_test : array, optional
        Feature matrix of test set. The default is None.

    Returns
    -------
    x_train, x_val, x_test : arrays
        normalized feature sets based on the statistics of training features.

    '''

    min_max_norm = preprocessing.MinMaxScaler(feature_range=(0,1))
    min_max_norm.fit(x_train)
    x_train = min_max_norm.fit_transform(x_train)
    if x_val is not None:
        x_val = min_max_norm.transform(x_val)
    
    if x_test is not None:
        x_test = min_max_norm.transform(x_test)
        
    return x_train, x_val, x_test
    

def data_shuffling(feature_set, label_set, seed_val):
    '''
    shuffling the features and corresponding labels with the same order.

    Parameters
    ----------
    feature_set : array
        feature matrix containing all data.
    label_set : array
        label vector containing all data.
    seed_val : int
        seed value for random generation.

    Returns
    -------
    feature_set : array
        subject-wise reordered feature set.
    label_set : array
        subject-wise reordered label set.

    '''
    
    length = np.arange(label_set.shape[0])
    np.random.seed(seed_val)  
    np.random.shuffle(length)
    feature_set = feature_set[length]
    label_set = label_set[length]
    
    return feature_set, label_set

  
def get_stats(metric):
    '''
    Get mean and std of metrics

    Parameters
    ----------
    metric : list
        containing the metrics of different folds.

    Returns
    -------
    mean_metric : float
        average value of the metric.
    std_metric : float
        standard deviation of the metric.

    '''
    
    metric = np.array(metric)
    mean_metric = np.mean(metric)
    std_metric = np.std(metric)
    
    return mean_metric, std_metric

def cross_val_stats(fold_stats):
    '''
    mean/std of metrics for cross validation experiments

    Parameters
    ----------
    fold_stats : dict
        evaluation metrics for each folds are kept in the dict.

    Returns
    -------
    four average values and four standard deviation values for four metrics 
    including accuracy, sensitivity, specifity, and auroc.

    '''

    acc = []
    sen = []
    spc = []
    auc = []
    for keys, vals in fold_stats.items():
        if type(vals) == dict:
            acc.append(vals['val_accuracy'])
            sen.append(vals['val_sensitivity'])
            spc.append(vals['val_specificity'])
            auc.append(vals['val_auc'])
    acc = np.array(acc)
    sen = np.array(sen)
    spc = np.array(spc)
    auc = np.array(auc)

    mean_acc, std_acc = get_stats(acc)
    mean_sen, std_sen = get_stats(sen)
    mean_spc, std_spc = get_stats(spc)
    mean_auc, std_auc = get_stats(auc)
    
    return mean_acc, mean_sen, mean_spc, mean_auc, std_acc, std_sen, std_spc, std_auc 




  

def feature_set_details(feature_set):
    feature_list = []
    for sequence,features in feature_set.items():
        features_names = list(features.keys()) # == list(features_df.column)
        features_names = features_names[3:]       # getting the feature names
        subject_ids =  list(features[features.columns[1]]) # get the subject ids
        subject_label = np.asarray(list(features[features.columns[2]])) # get the subject labels
        feature_values = features.values  # get the feature values
        feature_values = feature_values[:,3:] # the first 3 columns contain order, id, lbels 
        feature_values = feature_values.astype(np.float32)
        feature_list.append(feature_values)
    
    return feature_list, subject_ids, subject_label
    

def feature_squeeze(feature_list):
    
    feature_arrays = np.array(feature_list)
    mean_features = np.mean(feature_arrays, axis=0)
    sum_features = np.sum(feature_arrays, axis=0)
    std_features = np.std(feature_arrays, axis=0)
    
    delta_features = np.absolute(np.subtract(feature_arrays, mean_features))
    feature_set_squeeze = np.mean(delta_features, axis=0)
    
    return mean_features, sum_features, std_features, feature_set_squeeze


def normalize_Zscore(feature_set):
    set_mean = (np.vstack(np.mean(feature_set, axis=0))).T
    set_std = (np.vstack(np.std(feature_set, axis=0))).T
    set_normalized = (feature_set-set_mean)/set_std
    
    return set_mean, set_std, set_normalized




def print_summary(summary_file):
    for key, val in summary_file.items():
            temp_name = []
            temp_val = []
            for key_in, val_in in val.items():
                if val_in>=0:
                    temp_val.append(val_in)
                    temp_name.append(key_in)
                else:
                    pass
            mean_val = sum(temp_val)/len(temp_val)    
            max_val = max(temp_val)
            max_ind = temp_val.index(max_val)
            get_key = temp_name[max_ind]
            
            print('Method {} : Mean is {}, highest {} from seed {}'.format(
                key, mean_val,max_val, get_key))