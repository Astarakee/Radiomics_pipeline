>># **Radiomics Pipeline for Oncological Image Analysis**

This repository contains the implementation codes of radiomics analysis. <br>
This repo was developed for the study of 'A comparative study of radiomics and deep-learning based methods for pulmonary nodule malignancy prediction in low dose CT images.' <br>
If you use any of our code, please cite:

@article{Astaraki2021,
>  title = { A comparative study of radiomics and deep-learning based methods for pulmonary nodule malignancy prediction in low dose CT images }, <br> author = {Astaraki, Mehdi and Yang, Guang and Zakko, Yousuf and Toma Dasu, Iuliana and Smedby, Örjan  and Wang, Chunliang}, <br>
  url = {---}, <br>
  year = {2021} <br>
}

## <i>Background<i/> <br>

Since 2010, the field of imaging biomarkers has been more formalized under the name of 'radiomics'. <br> This term emanates from the words 'radio' that adopted from radiology, followed by the suffix 'omics'  which refers to the study of biological phenomena.<br>
<br>
The goal of the rapidly growing radiomics in the oncology field is to capture valuable information regarding the shape, size, intensity, and texture of tumor phenotype that would distinct or complement other sources of information such as clinical reports, and laboratory examinations. <br>
<br>
Considering the growing role of radiomics in oncology image studies, the __[Image Biomarker Standard Initiative (IBSI)](https://pubs.rsna.org/doi/10.1148/radiol.2020191145)__ has been established to provide a step by step guideline for radiomics analysis including feature nomenclature and definition, feature calculation, benchmark data set, and reporting regulations <br>
<br>
The goal of radiomic feature analysis is to develop a mathematical model or a function to stratify patients based on the predicted outcome using extracted radiomic features. From the machine learning perspective, this task is equivalent to develop a learning algorithm to predict the outcomes of given data points. In other word, the learning algorithm analyzes the training data and learns the underlying representative characteristics of the data to infer a hypothesis in order to predict the outcomes of unseen data. <br>
<br>
In current research practices, radiomic pipelines are usually developed manually with a specific dataset of that study.<br> Therefore, the performance and functionality of different pipeline settings depend on the properties of the employed learning algorithms and training procedures, which in turn would result in deviations of the model performance on the same dataset. <br> **In this repo, we tried to provide a standard radiomic analysis pipeline based on the classical machine learning algorithms.** 

## <i>Folder structure<i/>

<br>
├ <b>radiomics_pipeline</b><br>
│   ├── <b>data_loader</b> '<i>loading image data of extracted features</i>' <br>
│   │   ├── feature_loader.py<br>
│   │   ├── paths_and_dirs.py<br>
│   │   └── sitk_stuff.py<br>
│   ├── <b>experiments_results</b> '<i> dir for saving the summary of the experiments</i>' <br>
│   │   └── experiment_210613-1834.json<br>
│   ├── <b>feature_extraction</b> '<i> extracting the radiomics features</i>'<br>
│   │   ├── extraction_lung_nodule.py<br>
│   │   └── params.yaml<br>
│   ├── <b>features</b>  '<i> dir to save the radiomic features</i>' <br>
│   │   └── radiomic_features.csv<br>
│   ├── <b>feature_selections</b> '<i> feature selection methods</i>'<br>
│   │   ├── feature_selectors.py<br>
│   ├── <b>learning_algorithms</b> '<i> learning algorithm models</i>'<br>
│   │   ├── models.py<br>
│   ├── <b>preprocessing</b> '<i> preprocessing the features or images</i>'<br>
│   │   ├── class_balancing.py<br>
│   ├── <b>readme.ipynb</b> <br>
│   ├── <b>run</b> '<i> executing the experiments</i>' <br>
│   │   ├── main.py<br>
│   │   └── main_sfs.py<br>
│   ├── <b>training</b> '<i> trainer functions for learning algorithms</i>'<br>
│   │   └── trainer.py<br>
│   └── <b>utilities</b> '<i> some utility functions</i> '<br>
│       ├── feature_tools.py<br>
│       ├── json_stuff.py<br>
│       ├── metircs_evaluation.py<br>
│       └── ...<br>


## <i>Usage<i/>
The original study was conducted on the __[Kaggle Data Science Bowl 2017 dataset](https://www.kaggle.com/c/data-science-bowl-2017)__. <br>To get access to the segmentation mask files, please contact me via "mehdi.astaraki@sth.kth.se"

## <i>Requirements<i/>
<br>
- Python >= 3.6 <br><br>
All packages used in this repo are listed in the 'requirements.txt'. <br>
Run 'pip3 install -r requirements.txt' to install the required packages. 
