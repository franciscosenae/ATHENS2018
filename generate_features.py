"""
Load the data, generate features and save the resulting dataframe to csv
"""
import os

import pandas as pd

from utils import *



def main():
    dfs = []
    for y in [2016, 2017, 2018]:
        data_path = os.path.join('data/raw_from_matlab', f'data{y}.mat')
        print(f'Loading {data_path}')
        _df = load_dataframe(data_path)
        save_path = f'data/processed/{y}.csv'
        print(f'Saving data with features to {save_path}')
        _df.to_csv(save_path, index=False)
        dfs.append(_df)

    full_df = pd.concat(dfs)
    save_path = 'data/processed/all.csv'
    print(f'Saving full data with features to {save_path}')
    full_df.to_csv(save_path, index=False)


if __name__ == '__main__':
    main()
