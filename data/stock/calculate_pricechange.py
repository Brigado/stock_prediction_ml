import os
import pandas as pd
import numpy as np

source_path_dir = "Stock/"
target_path_dir = "stock_development/"


for company_file in os.listdir(source_path_dir):
    #print(company_file)
    df = pd.read_csv(source_path_dir + company_file)
    df['returns'] = 0.0
    df['change'] = 0.0
    df['development'] = 0
    for index, row in df.iterrows():
        if index != 0:
            df.at[index, 'returns'] = (df.at[index, 'Close']/df.at[index-1, 'Close'])-1
            df.at[index, 'change'] = (df.at[index, 'Close'] - df.at[index-1, 'Close'])
    df['zscore'] = (df['returns'] - df['returns'].mean()) / df['returns'].std(ddof=0)
    df['volatile'] = np.where((df['zscore']>1) | (df['zscore']<-1), 1, 0)
    # if volatile && change>0: development = 1 elif volatile && change < 0: development = -1 else: development = 0
    df['development'] = np.where(df['change']>0, 1, -1)
    df.loc[df.volatile == 0, 'development'] = 0
    df = df[['Date', 'development']]
    #print(df.head())
    df.to_csv(target_path_dir+"development_"+company_file)

