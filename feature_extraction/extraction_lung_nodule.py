import os
import sys
import six
import glob
import pandas as pd
from radiomics import featureextractor
sys.path.append('../../radiomics_pipeline/')
from data_loader.sitk_stuff import read_nifti
from data_loader.paths_and_dirs import get_filepath, match_img_label, read_csv_file


radiomic_feature_name = 'radiomic_features_test.csv'

lung_nodule_csv_path = '/media/mehdi/KTH/DeepLearning/DataRepository/Lung Nodule/data/Lung_CT/'
data_path = '/media/mehdi/KTH/DeepLearning/DataRepository/Lung Nodule/data/Lung_CT/train/'


radiomic_path_write = os.path.join('../features/', radiomic_feature_name)
csv_path = glob.glob(lung_nodule_csv_path + "*.csv")
csv_contents = read_csv_file(csv_path[0])
img_path = get_filepath(data_path, 'cropped')
mask_path = get_filepath(data_path, 'mask')
samples_img = match_img_label(img_path, csv_contents)
samples_msk = match_img_label(mask_path, csv_contents)
data_length = len(samples_img)


# Instantiating Radiomics Feature Extraction
extractor = featureextractor.RadiomicsFeatureExtractor()
param_path = os.path.join(os.getcwd(), 'params.yaml')
extractor = featureextractor.RadiomicsFeatureExtractor(param_path)
print('Extraction parameters:\n\t', extractor.settings)
print('Enabled filters:\n\t', extractor.enabledImagetypes)
print('Enabled features:\n\t', extractor.enabledFeatures)



for ind in range(len(samples_img)):
    
    img_path = samples_img[ind][0]
    mask_path = samples_msk[ind][0]
    subject_name = os.path.split(img_path)[-1]
    subject_label = samples_img[ind][1]
    
    print('\n'*10)
    print('working on case {} out of {}:'.format(ind+1, data_length))
    print('\n'*10)
    
    
    img_itk, img_size, img_spacing, _, _ = read_nifti(img_path)
    mask_itk, mask_size, mask_spacing, _, _ = read_nifti(mask_path)
    
    assert img_size == mask_size and img_spacing == mask_spacing, "image and mask should have the same size and spacing!"
    features = extractor.execute(img_itk, mask_itk)
    features_all = {}
    for key, value in six.iteritems(features):
        if key.startswith('original') or key.startswith('wavelet') or \
        key.startswith('log'):
            features_all['Subject_ID'] = subject_name
            features_all['Subject_Label'] = subject_label
            features_all[key] = features[key]
            
    df = pd.DataFrame(data=features_all,  index=[ind])
    if ind == 0:
        df.to_csv(radiomic_path_write, mode='a')
    else:
        df.to_csv(radiomic_path_write, header = None, mode='a')
        

