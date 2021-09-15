import os
import json


def save_json(write_dir, filename, dict_summary):
    
    json_dir = os.path.join(write_dir, filename)
    with open(json_dir, 'w') as fp:
        json.dump(dict_summary, fp, indent = 4)
        
    

def prepare_summary(path_to_radiomics, model_name, selector_name, seed_val,
                         n_fold_split, balancing, clf_params, fold_stats):
    
    report_summary = {}
    report_summary['feature_path'] = path_to_radiomics
    report_summary['model_name'] = model_name
    report_summary['selector_name'] = selector_name
    report_summary['seed_value'] = seed_val
    report_summary['n_fold_cv'] = n_fold_split
    report_summary['class_balancing'] = balancing
    report_summary['model_config'] = clf_params
    report_summary['cross_val_metrics'] = fold_stats
    
    return report_summary