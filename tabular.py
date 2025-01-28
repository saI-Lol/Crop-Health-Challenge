import pandas as pd
import numpy as np
import argparse
from joblib import Parallel, delayed
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


def main(args):    
    data_path = args.data_path
    splitter = args.splitter
    n_splits = args.n_splits
    random_state = args.random_state
    group_col = args.group_col
    seeds = set()
    np.random.seed(random_state)
    while len(seeds) < args.num_seeds:
        seed = np.random.randint(0, 1000)
        seeds.add(seed)        

    data = pd.read_csv(data_path)
    data['geometry'] = data['geometry'].apply(loads)
    data[['min_x', 'min_y', 'max_x', 'max_y']] = data.apply(
        lambda row: pd.Series(row['geometry'].bounds),
        axis=1
    )

    new_features = Parallel(n_jobs=-1)(delayed(process_row_for_features)(index, row)
                                    for index, row in tqdm(data.iterrows(), total=len(data)))
    new_features_df = pd.DataFrame(new_features).set_index('index')
    imputer = SimpleImputer(strategy="mean")
    new_features_df = pd.DataFrame(imputer.fit_transform(new_features_df), columns=new_features_df.columns)
    data = data.join(new_features_df)

    to_drop = ['geometry', 'tif_path', 'FarmID', "State"]
    data_combined = data.drop(columns=to_drop).copy()

    category_mapper = {label: idx for idx, label in enumerate(data_combined['category'].unique()) if pd.notna(label)}
    idx_to_category_mapper = {idx: label for idx, label in enumerate(data_combined['category'].unique()) if pd.notna(label)}
    data_combined['target'] = data_combined['category'].map(category_mapper)
    data_combined = data_combined.drop(columns=['category'], axis=1)

    # crops = np.unique(data[['Crop', 'CLast', 'CNext']].values)
    # crop_idx_map = {crop:idx for idx, crop in enumerate(crops)}
    # data_combined['Crop'] = data_combined['Crop'].map(crop_idx_map)
    # data_combined['CLast'] = data_combined['CLast'].map(crop_idx_map)
    # data_combined['CNext'] = data_combined['CNext'].map(crop_idx_map)
    # data_combined['CNextMinusCrop'] = data_combined['CNext'] - data_combined['Crop']
    # data_combined['CropMinusCLast'] = data_combined['Crop'] - data_combined['CLast']

    cols = ['ExpYield', 'WaterCov', 'IrriCount', 'CHeight', 'CropCoveredArea']
    data_combined[cols] = data_combined[cols].astype(str)

    categorical_cols = data_combined.select_dtypes(include=['object']).drop(['dataset'], axis=1).columns
    print(categorical_cols)

    train_encoded = data_combined[data_combined['dataset'] == 'train'].reset_index(drop=True)
    test_encoded = data_combined[data_combined['dataset'] == 'test'].reset_index(drop=True)
    X = train_encoded.drop(columns=['target', 'dataset'])
    y = train_encoded['target'].astype(int)
    
    if splitter == "tts":
        for seed in seeds:
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)
            model = LGBMClassifier(verbose=-1, random_state=seed)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)
            f1 = f1_score(y_valid, y_pred, average='weighted')
            print(f"\nWeighted F1 Score for seed {seed}:", f1)

    elif splitter == "skf":
        for seed in seeds:
            skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
            results = {'y_true': [], 'y_pred': []}
            for train_idx, valid_idx in skf.split(X, y):
                X_train, X_valid = X.loc[train_idx], X.loc[valid_idx]
                y_train, y_valid = y.loc[train_idx], y.loc[valid_idx]
                train = pd.concat([X_train, y_train], axis=1)

                for col in categorical_cols:
                    mean_dict = train.groupby(col)['target'].mean().to_dict()
                    median_dict = train.groupby(col)['target'].median().to_dict()
                    # std_dict = train.groupby(col)['target'].std().to_dict()
                    min_dict = train.groupby(col)['target'].min().to_dict()
                    max_dict = train.groupby(col)['target'].max().to_dict()
                    nunique_dict = train.groupby(col)['target'].nunique().to_dict()
                    X_valid[f"{col}_mean_target"] = X_valid[col].map(mean_dict)
                    X_valid[f"{col}_median_target"] = X_valid[col].map(median_dict)
                    # X_valid[f"{col}_std_target"] = X_valid[col].map(std_dict)
                    # X_valid[f"{col}_min_target"] = X_valid[col].map(min_dict)
                    # X_valid[f"{col}_max_target"] = X_valid[col].map(max_dict)
                    # X_valid[f"{col}_nunique_target"] = X_valid[col].map(nunique_dict)

                    X_train[f"{col}_mean_target"] = X_train[col].map(mean_dict)
                    X_train[f"{col}_median_target"] = X_train[col].map(median_dict)
                    # X_train[f"{col}_std_target"] = X_train[col].map(std_dict)
                    # X_train[f"{col}_min_target"] = X_train[col].map(min_dict)
                    # X_train[f"{col}_max_target"] = X_train[col].map(max_dict)
                    # X_train[f"{col}_nunique_target"] = X_train[col].map(nunique_dict)

                    X_valid[f"{col}_frequency_encoded"] = X_valid[col].map(Counter(X_valid[col]))
                    X_train[f"{col}_frequency_encoded"] = X_train[col].map(Counter(X_train[col]))

                    # encoder = LabelEncoder()
                    # encoder.fit(X_temp[col])
                    # X_valid[col] = encoder.transform(X_valid[col])
                    # X_train[col] = encoder.transform(X_train[col])
                    # X_train = X_train.drop(columns=[col], axis=1)
                    # X_valid = X_valid.drop(columns=[col], axis=1)

                X_train['dataset'] = 'train'
                X_valid['dataset'] = 'valid'
                X_temp = pd.concat([X_train, X_valid])
                X_temp = pd.get_dummies(X_temp, columns=categorical_cols, dtype='int32')
                X_train = X_temp[X_temp['dataset'] == 'train'].drop(columns=['dataset'], axis=1)
                X_valid = X_temp[X_temp['dataset'] == 'valid'].drop(columns=['dataset'], axis=1)

                #get min and max values in dfs
                # print(X_train.min().min(), X_train.max().max())
                # print(X_valid.min().min(), X_valid.max().max())
                    

                model = LGBMClassifier(verbose=-1, random_state=seed)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_valid)
                results['y_true'].extend(y_valid)
                results['y_pred'].extend(y_pred)
            f1 = f1_score(results['y_true'], results['y_pred'], average='weighted')
            print(f"\nWeighted F1 Score for seed {seed}:", f1)

    elif splitter == "gkf":
        for seed in seeds:
            gkf = GroupKFold(n_splits=n_splits)
            results = {'y_true': [], 'y_pred': []}
            for train_idx, valid_idx in gkf.split(X, y, groups=X[group_col]):
                X_train, X_valid = X.loc[train_idx], X.loc[valid_idx]
                y_train, y_valid = y.loc[train_idx], y.loc[valid_idx]
                model = LGBMClassifier(verbose=-1, random_state=seed)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_valid)
                results['y_true'].extend(y_valid)
                results['y_pred'].extend(y_pred)
            f1 = f1_score(results['y_true'], results['y_pred'], average='weighted')
            print(f"\nWeighted F1 Score for seed {seed}:", f1)

    elif splitter == "kf":        
        for seed in seeds:
            kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
            results = {'y_true': [], 'y_pred': []}
            for train_idx, valid_idx in kf.split(X):
                X_train, X_valid = X.loc[train_idx], X.loc[valid_idx]
                y_train, y_valid = y.loc[train_idx], y.loc[valid_idx]
                model = LGBMClassifier(verbose=-1, random_state=seed)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_valid)
                results['y_true'].extend(y_valid)
                results['y_pred'].extend(y_pred)
            f1 = f1_score(results['y_true'], results['y_pred'], average='weighted')
            print(f"\nWeighted F1 Score for seed {seed}:", f1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Telangana Crop Health Challenge')
    parser.add_argument('--data_path', type=str, default="/kaggle/input/sentineltimeseriesdata/SentinelTimeSeriesData/Data.csv", help='Path to the data CSV file')
    parser.add_argument("--splitter", type=str, default="tts", help="Splitter to use for cross-validation")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of splits for cross-validation")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    parser.add_argument("--group_col", type=str, default="Crop", help="Column to use for GroupKFold")
    parser.add_argument("--num_seeds", type=int, default = 1, help="Number of seeds to use for cross-validation")
    args = parser.parse_args()
    if args.splitter not in ["skf", "gkf", "tts", "kf"]:
        raise ValueError("Splitter should be one of ['skf', 'gkf', 'tts', 'kf']")
    if args.splitter in ["skf", "gkf", "kf"] and args.n_splits is None:
        raise ValueError("Number of splits should be provided")    
    if args.splitter in ["gkf"] and args.group_col is None:
        raise ValueError("Group column should be provided for GroupKFold")
    main(args)