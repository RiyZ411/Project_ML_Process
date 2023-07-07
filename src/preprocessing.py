import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import util as utils
from imblearn.over_sampling import SMOTE

def load_dataset():
    config = utils.load_config()
    dataset = pd.read_csv(config["dataset_path"])
    return dataset

def remove_missing(dataset):
    #missing
    #definisikan nilai missing yang kemungkinan terjadi
    missing_values = ['', ' ', 'NaN', 'Nan', 'nan', '.', ',','---']
    col_names = list(dataset.columns)
    dataset[col_names] = dataset[col_names].replace(missing_values, np.nan)
    print(dataset.isna().sum())
    dataset = dataset.dropna()
    return dataset.isna().sum()

def feature_engineering(dataset):
    dataset = dataset.drop(['tanggal'], axis=1)
    return dataset

def label_encoding(dataset):
    label_encoder = preprocessing.LabelEncoder()

    #kolom stasiun
    dataset['stasiun']= label_encoder.fit_transform(dataset['stasiun'])

    #kolom critical
    dataset['critical']= label_encoder.fit_transform(dataset['critical'])

    #kolom categori
    dataset['categori'] = dataset['categori'].replace(['BAIK', 'TIDAK SEHAT'],[1, 0])

    return dataset
        
def balance_data(dataset):
    sm = SMOTE(random_state = 42)
    X_res, y_res = sm.fit_resample(dataset.iloc[:,:-1], dataset.iloc[:,-1:])
    dataset = pd.concat([X_res, y_res], axis=1)
    return dataset

def drop_duplicate(dataset):
    dataset.duplicated().sum()
    dataset = dataset.drop_duplicates()
    return dataset

def split_and_dump(dataset):
    config = utils.load_config()
    x = dataset[config["predictors"]].copy()
    y = dataset[config["label"]].copy()

    #split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42, stratify = y)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = 0.3, random_state = 42, stratify = y_train)
    
    #dump
    utils.pickle_dump(dataset, config["dataset_cleaned_path"])
    utils.pickle_dump(x_train, config["train_set_path"][0])
    utils.pickle_dump(y_train, config["train_set_path"][1])
    utils.pickle_dump(x_valid, config["valid_set_path"][0])
    utils.pickle_dump(y_valid, config["valid_set_path"][1])
    utils.pickle_dump(x_test, config["test_set_path"][0])
    utils.pickle_dump(y_test, config["test_set_path"][1])

if __name__ == "__main__":
    # 1. Load dataset
    dataset = load_dataset()

    # 2. removing missing
    missing = remove_missing(dataset)

    # 3. feature engineering
    feature_eng = feature_engineering(missing)

    # 4. lebel encoding
    label_enc = label_encoding(feature_eng)

    # 5. balancing data
    balancing =  balance_data(label_enc)

    #6. drop duplicate
    duplicate = drop_duplicate(balancing)

    # 7. split and dump set data
    splitndump = split_and_dump(duplicate)


    