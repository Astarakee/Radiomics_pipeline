import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.append('../../radiomics_pipeline/')

from datetime import datetime
from training.trainer import learning
from learning_algorithms.models import decicion_tree
from feature_selections.feature_selectors import lasso
from sklearn.model_selection import KFold
from utilities.json_stuff import save_json, prepare_summary
from data_loader.feature_loader import load_radiomic_set
from preprocessing.class_balancing import smote_balancing
from utilities.feature_tools import data_shuffling, cross_val_stats
from utilities.feature_tools import feature_normalization, train_val_split

seed_val = 42
n_fold_split = 5
balancing = True
feature_selection = True

if feature_selection:
    selector_name = lasso.__name__
else:
    selector_name = None   
model_name  = decicion_tree.__name__

time = datetime.now().strftime("%y%m%d-%H%M")
exp_name_json = 'experiment_'+time+'.json'




path_to_radiomics = '../features/radiomic_features.csv'
subject_ids ,features_names, feature_set, label_set = load_radiomic_set(path_to_radiomics)


if balancing:    
    feature_set, label_set = smote_balancing(feature_set, label_set)


feature_set, label_set = data_shuffling(feature_set, label_set, seed_val)



fold_num = 0
kf = KFold(n_splits = n_fold_split, shuffle = False) 
fold_stats = {}
for train_index, test_index in kf.split(label_set):
    fold_num += 1
    fold_name = 'fold_'+str(fold_num)
    print('Working on fold: {}'.format(fold_num))
    
    x_train, x_val, y_train, y_val = train_val_split(
        feature_set, label_set, train_index, test_index)
    
    x_train, x_val, _ = feature_normalization(x_train, x_val, x_test = None)
    
    if feature_selection:
        
        feature_selector = lasso(x_train, y_train, x_val, x_test=None, n_fold=5, max_iters=50, thr=0.5)
        x_train = feature_selector['train']
        x_val = feature_selector['val']
        feature_indices = feature_selector['feature_indices']
        fold_name_features = fold_name+'_selected_features'
        fold_stats[fold_name_features] = feature_indices
        

    clf = decicion_tree()
    clf_summary = learning(clf, x_train, y_train, x_val, y_val, x_test = None)
    fold_stats[fold_name] = clf_summary

clf_params = clf.get_params()
mean_acc, mean_sen, mean_spc, mean_auc, _, _, _, _ = cross_val_stats(fold_stats)
print('the average auc value of {}fold cross validation is: {}'.format(n_fold_split, mean_auc))



report_summary = prepare_summary(path_to_radiomics, model_name,
                                 selector_name, seed_val,
                                 n_fold_split, balancing,
                                 clf_params, fold_stats)    
save_json('../experiments_results/', exp_name_json, report_summary)
    