import numpy as np
import pandas as pd



def load_radiomic_set(feature_path):
    '''
    loading radiomics features saved in a .csv file
    Parameters
    ----------
    feature_path : str
        full path to the .csv file.

    Returns
    -------
    subject_ids : list
    feature_values : array
    label_values : array

    '''
    
    
    features_df = pd.read_csv(feature_path)    
    features_names = list(features_df.keys()) # == list(features_df.column)
    features_names = features_names[3:]       # getting the feature names
    subject_ids =  list(features_df[features_df.columns[1]]) # get the subject ids
    label_values = np.asarray(list(features_df[features_df.columns[2]])) # get the subject labels
    feature_values = features_df.values  # get the feature values
    feature_values = feature_values[:,3:] # the first 3 columns contain order, id, labels 
    feature_values = feature_values.astype(np.float32)
    
    return subject_ids ,features_names, feature_values, label_values