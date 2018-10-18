import os
import pandas as pd
import numpy as np

source_path_dir = "Stock/"
target_path_dir = "Stock_is_volatile/"


for company_file in os.listdir(source_path_dir):
    #print(company_file)
    df = pd.read_csv(source_path_dir + company_file)
    df['returns'] = 0.0
    for index, row in df.iterrows():
        if index != 0:
            df.at[index, 'returns'] = (df.at[index, 'Close']/df.at[index-1, 'Close'])-1
    df['zscore'] = (df['returns'] - df['returns'].mean()) / df['returns'].std(ddof=0)
    df['volatile'] = np.where((df['zscore']>1) | (df['zscore']<-1), 1, 0)
    df = df[['Date', 'volatile']]
    #print(df.head())
    df.to_csv(target_path_dir+"is_volatile_"+company_file)