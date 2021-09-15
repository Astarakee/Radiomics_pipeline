import numpy as np
from sklearn.metrics import confusion_matrix


def conf_matrix(y_true, y_pred):
    
    target_labels = np.array(y_true)
    predictions = np.array(y_pred)
    matrix = confusion_matrix(target_labels, predictions)
    
    return matrix


def metrics(conf_matrix):
    
    tp = conf_matrix[1][1]
    tn = conf_matrix[0][0]
    fp = conf_matrix[0][1]
    fn = conf_matrix[1][0]
    
    accuracy = (float (tp+tn) / float(tp + tn + fp + fn))
    sensitivity = (tp / float(tp + fn))
    specificity = (tn / float(tn + fp))

    return accuracy, sensitivity, specificity
