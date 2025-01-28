import pandas as pd
import numpy as np
import argparse
# from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold, KFold
from helper import process_row_for_features
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from shapely.wkt import loads
from itertools import combinations
import wandb

def encode_data(X_train, X_valid, y_train, y_valid, encs):
    X_train_encoded_list = []
    X_valid_encoded_list = []
    for enc in encs:
        if enc == "frequency":
            for col in X_train.columns:
                freq = X_train[col].value_counts().to_dict()
                X_train_encoded_list.append(X_train[col].map(freq).rename(col + "_freq"))
                X_valid_encoded_list.append(X_valid[col].map(freq).rename(col + "_freq"))
        if enc == "label": 
            X_train_temp = X_train.copy()   
            X_valid_temp = X_valid.copy()       
            X_train_temp['dataset'] = 'train'
            X_valid_temp['dataset'] = 'valid'
            X = pd.concat([X_train_temp, X_valid_temp], axis=0)
            for col in X_train.columns:
                le = LabelEncoder()                
                X[col] = le.fit_transform(X[col])
            train = X[X['dataset'] == 'train'].drop('dataset', axis=1)
            valid = X[X['dataset'] == 'valid'].drop('dataset', axis=1)
            for col in X_train.columns:
                X_train_encoded_list.append(train[col].rename(col + "_label"))
                X_valid_encoded_list.append(valid[col].rename(col + "_label"))
        
        train_temp = pd.concat([X_train, y_train], axis=1)
        if enc == "target_mean":
            for col in X_train.columns:
                target_mean = train_temp.groupby(col)['target'].mean().to_dict()
                X_train_encoded_list.append(X_train[col].map(target_mean).rename(col + "_target_mean"))
                X_valid_encoded_list.append(X_valid[col].map(target_mean).rename(col + "_target_mean"))
        if enc == "target_median":
            for col in X_train.columns:
                target_median = train_temp.groupby(col)['target'].median().to_dict()
                X_train_encoded_list.append(X_train[col].map(target_median).rename(col + "_target_median"))
                X_valid_encoded_list.append(X_valid[col].map(target_median).rename(col + "_target_median"))
        if enc == "target_min":
            for col in X_train.columns:
                target_min = train_temp.groupby(col)['target'].min().to_dict()
                X_train_encoded_list.append(X_train[col].map(target_min).rename(col + "_target_min"))
                X_valid_encoded_list.append(X_valid[col].map(target_min).rename(col + "_target_min"))
        if enc == "target_max":
            for col in X_train.columns:
                target_max = train_temp.groupby(col)['target'].max().to_dict()
                X_train_encoded_list.append(X_train[col].map(target_max).rename(col + "_target_max"))
                X_valid_encoded_list.append(X_valid[col].map(target_max).rename(col + "_target_max"))
        if enc == "target_std":
            for col in X_train.columns:
                target_std = train_temp.groupby(col)['target'].std().to_dict()
                X_train_encoded_list.append(X_train[col].map(target_std).rename(col + "_target_std"))
                X_valid_encoded_list.append(X_valid[col].map(target_std).rename(col + "_target_std"))            
            
    
    X_train_encoded = pd.concat(X_train_encoded_list, axis=1)
    X_valid_encoded = pd.concat(X_valid_encoded_list, axis=1)
    return X_train_encoded, X_valid_encoded, y_train, y_valid



def main(args):    
    # wandb.init(project="telangana-crop-health", name="Experiment-1")
    data_path = args.data_path
    # splitter = args.splitter
    n_splits = args.n_splits
    random_state = args.random_state
    # group_col = args.group_col
    # seeds = set()
    # np.random.seed(random_state)
    # while len(seeds) < args.num_seeds:
    #     seed = np.random.randint(0, 1000)
    #     seeds.add(seed)        

    data = pd.read_csv(data_path)
    data = data[data['dataset'] == 'train'].reset_index(drop=True)
    # data['geometry'] = data['geometry'].apply(loads)
    # data[['min_x', 'min_y', 'max_x', 'max_y']] = data.apply(
    #     lambda row: pd.Series(row['geometry'].bounds),
    #     axis=1
    # )

    # new_features = Parallel(n_jobs=-1)(delayed(process_row_for_features)(index, row)
    #                                 for index, row in tqdm(data.iterrows(), total=len(data)))
    # new_features_df = pd.DataFrame(new_features).set_index('index')
    # imputer = SimpleImputer(strategy="mean")
    # new_features_df = pd.DataFrame(imputer.fit_transform(new_features_df), columns=new_features_df.columns)
    # data = data.join(new_features_df)

    to_drop = ['geometry', 'tif_path', 'FarmID', "State", "SDate", "HDate", "dataset"]
    data_combined = data.drop(columns=to_drop)

    category_mapper = {label: idx for idx, label in enumerate(data_combined['category'].unique()) if pd.notna(label)}
    idx_to_category_mapper = {idx: label for idx, label in enumerate(data_combined['category'].unique()) if pd.notna(label)}
    data_combined['target'] = data_combined['category'].map(category_mapper)
    data_combined = data_combined.drop(columns=['category'], axis=1)

    categorical_cols = ['Crop', 'District', 'Sub-District', 'CropCoveredArea', 
    'CNext', 'CLast', 'CTransp', 'IrriType', 
    'IrriSource', 'IrriCount', 'WaterCov', 'ExpYield', 'Season']
    encoding = ["frequency", "label", "target_mean", "target_median", "target_std", "target_min", "target_max"]
    data_combined = data_combined[categorical_cols + ['target']].copy()

    column_combinations = []
    encoding_combinations = []
    for i in range(2, len(categorical_cols)+1):
        column_combinations.extend(combinations(categorical_cols, i))
    
    for i in range(1, len(encoding)+1):
        encoding_combinations.extend(combinations(encoding, i))
    
    for cols in column_combinations:
        cols = list(cols)
        df_cols = data_combined[cols + ['target']].copy()
        if len(cols) > 1:
            df_cols['combined'] = df_cols[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        cols = list(df_cols.columns)
        
        for encs in encoding_combinations:
            encs = list(encs)
            df = df_cols.copy()
            X = df.drop(columns=['target'], axis=1)
            y = df['target']
            skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
            results = {'y_true': [], 'y_pred': []}
            for train_idx, valid_idx in skf.split(X, y):
                X_train, X_valid = X.loc[train_idx], X.loc[valid_idx]
                y_train, y_valid = y.loc[train_idx], y.loc[valid_idx]
                X_train_encoded, X_valid_encoded, y_train, y_valid = encode_data(X_train, X_valid, y_train, y_valid, encs)
                model = LGBMClassifier(verbose=-1, random_state=random_state)
                model.fit(X_train_encoded, y_train)
                y_pred = model.predict(X_valid_encoded)
                results['y_true'].extend(y_valid)
                results['y_pred'].extend(y_pred)
            f1 = f1_score(results['y_true'], results['y_pred'], average='weighted')
            print(f"Features: {cols}\nEncoding: {encs}\nF1 Score: {f1}\n")
            # wandb.log({
            #     "features": cols,
            #     "encoding": encs,
            #     "f1": f1
            # })






    # crops = np.unique(data[['Crop', 'CLast', 'CNext']].values)
    # crop_idx_map = {crop:idx for idx, crop in enumerate(crops)}
    # data_combined['Crop'] = data_combined['Crop'].map(crop_idx_map)
    # data_combined['CLast'] = data_combined['CLast'].map(crop_idx_map)
    # data_combined['CNext'] = data_combined['CNext'].map(crop_idx_map)
    # data_combined['CNextMinusCrop'] = data_combined['CNext'] - data_combined['Crop']
    # data_combined['CropMinusCLast'] = data_combined['Crop'] - data_combined['CLast']

    # cols = ['ExpYield', 'WaterCov', 'IrriCount', 'CHeight', 'CropCoveredArea']
    # data_combined[cols] = data_combined[cols].astype(str)

    # categorical_cols = data_combined.select_dtypes(include=['object']).drop(['dataset'], axis=1).columns
    # for col in categorical_cols:
    #     le = LabelEncoder()
    #     data_combined[col] = le.fit_transform(data_combined[col])


    # train_encoded = data_combined[data_combined['dataset'] == 'train'].reset_index(drop=True)
    # test_encoded = data_combined[data_combined['dataset'] == 'test'].reset_index(drop=True)
    # X = train_encoded.drop(columns=['target', 'dataset'])
    # y = train_encoded['target'].astype(int)
    
    # if splitter == "tts":
    #     for seed in seeds:
    #         X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)
    #         model = LGBMClassifier(verbose=-1, random_state=seed)
    #         model.fit(X_train, y_train)
    #         y_pred = model.predict(X_valid)
    #         f1 = f1_score(y_valid, y_pred, average='weighted')
    #         print(f"\nWeighted F1 Score for seed {seed}:", f1)

    # elif splitter == "skf":
        # for seed in seeds:
        #     skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
        #     results = {'y_true': [], 'y_pred': []}
        #     for train_idx, valid_idx in skf.split(X, y):
        #         X_train, X_valid = X.loc[train_idx], X.loc[valid_idx]
        #         y_train, y_valid = y.loc[train_idx], y.loc[valid_idx]                   
        #         model = LGBMClassifier(verbose=-1, random_state=seed)
        #         model.fit(X_train, y_train)
        #         y_pred = model.predict(X_valid)
        #         results['y_true'].extend(y_valid)
        #         results['y_pred'].extend(y_pred)
        #     f1 = f1_score(results['y_true'], results['y_pred'], average='weighted')
        #     print(f"\nWeighted F1 Score for seed {seed}:", f1)

    # elif splitter == "gkf":
    #     for seed in seeds:
    #         gkf = GroupKFold(n_splits=n_splits)
    #         results = {'y_true': [], 'y_pred': []}
    #         for train_idx, valid_idx in gkf.split(X, y, groups=X[group_col]):
    #             X_train, X_valid = X.loc[train_idx], X.loc[valid_idx]
    #             y_train, y_valid = y.loc[train_idx], y.loc[valid_idx]
    #             model = LGBMClassifier(verbose=-1, random_state=seed)
    #             model.fit(X_train, y_train)
    #             y_pred = model.predict(X_valid)
    #             results['y_true'].extend(y_valid)
    #             results['y_pred'].extend(y_pred)
    #         f1 = f1_score(results['y_true'], results['y_pred'], average='weighted')
    #         print(f"\nWeighted F1 Score for seed {seed}:", f1)

    # elif splitter == "kf":        
    #     for seed in seeds:
    #         kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    #         results = {'y_true': [], 'y_pred': []}
    #         for train_idx, valid_idx in kf.split(X):
    #             X_train, X_valid = X.loc[train_idx], X.loc[valid_idx]
    #             y_train, y_valid = y.loc[train_idx], y.loc[valid_idx]
    #             model = LGBMClassifier(verbose=-1, random_state=seed)
    #             model.fit(X_train, y_train)
    #             y_pred = model.predict(X_valid)
    #             results['y_true'].extend(y_valid)
    #             results['y_pred'].extend(y_pred)
    #         f1 = f1_score(results['y_true'], results['y_pred'], average='weighted')
    #         print(f"\nWeighted F1 Score for seed {seed}:", f1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Telangana Crop Health Challenge')
    parser.add_argument('--data_path', type=str, default="/kaggle/input/sentineltimeseriesdata/SentinelTimeSeriesData/Data.csv", help='Path to the data CSV file')
    # parser.add_argument("--splitter", type=str, default="tts", help="Splitter to use for cross-validation")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of splits for cross-validation")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    # parser.add_argument("--group_col", type=str, default="Crop", help="Column to use for GroupKFold")
    # parser.add_argument("--num_seeds", type=int, default = 1, help="Number of seeds to use for cross-validation")
    args = parser.parse_args()
    # if args.splitter not in ["skf", "gkf", "tts", "kf"]:
    #     raise ValueError("Splitter should be one of ['skf', 'gkf', 'tts', 'kf']")
    # if args.splitter in ["skf", "gkf", "kf"] and args.n_splits is None:
    #     raise ValueError("Number of splits should be provided")    
    # if args.splitter in ["gkf"] and args.group_col is None:
    #     raise ValueError("Group column should be provided for GroupKFold")
    main(args)