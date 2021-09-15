import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.append('../../radiomics_pipeline/')

from datetime import datetime
from training.trainer import learning, learning_with_sfs
from learning_algorithms.models import random_forest
from feature_selections.feature_selectors import sequential_selection
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
    selector_name = sequential_selection.__name__
else:
    selector_name = None   
model_name  = random_forest.__name__

time = datetime.now().strftime("%y%m%d-%H%M")
exp_name_json = 'experiment_'+time+'.json'


path_to_radiomics = '../features/radiomic_features.csv'
subject_ids ,features_names, feature_set, label_set = load_radiomic_set(path_to_radiomics)


if balancing:    
    feature_set, label_set = smote_balancing(feature_set, label_set)


feature_set, label_set = data_shuffling(feature_set, label_set, seed_val)


# step 1: finding the most informative subset of features with SFS

n_features = 2
clf_model = random_forest()
sfs_model = sequential_selection(clf_model, n_features=n_features, foward_state=True, floating_state=False,
                         metric='roc_auc', k_fold=3, n_jobs=-1)
feature_set_norm, _, _ = feature_normalization(feature_set, x_val = None, x_test = None)
print('Sequential feature selection is in progress ...')
sfs_summary = learning_with_sfs(sfs_model, feature_set_norm, label_set, n_features=n_features)
print('Done with sequential feature selection step!')
exp_name = str(n_features) + ' selected features_names'
selected_features = list(sfs_summary[exp_name])


# step 2: cross validation and prediction with only the selected features

feature_subset = feature_set[:, selected_features]

fold_num = 0
kf = KFold(n_splits = n_fold_split, shuffle = False) 
fold_stats = {}
for train_index, test_index in kf.split(label_set):
    fold_num += 1
    fold_name = 'fold_'+str(fold_num)
    print('Working on fold: {}'.format(fold_num))
    
    x_train, x_val, y_train, y_val = train_val_split(
        feature_subset, label_set, train_index, test_index)
    
    x_train, x_val, _ = feature_normalization(x_train, x_val, x_test = None)
            

    clf = random_forest(n_estimators=50, criterion='gini', max_depth=10, class_weight=None)
    clf_summary = learning(clf, x_train, y_train, x_val, y_val, x_test = None)
    fold_stats[fold_name] = clf_summary

fold_stats['sfs_selected_features'] = selected_features
clf_params = clf.get_params()
mean_acc, mean_sen, mean_spc, mean_auc, _, _, _, _ = cross_val_stats(fold_stats)
print('the average auc value of {}-fold cross validation is: {}'.format(n_fold_split, mean_auc))


report_summary = prepare_summary(path_to_radiomics, model_name,
                                 selector_name, seed_val,
                                 n_fold_split, balancing,
                                 clf_params, fold_stats)    
save_json('../experiments_results/', exp_name_json, report_summary)
    