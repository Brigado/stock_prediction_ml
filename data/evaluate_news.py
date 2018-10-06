import pandas as pd
import numpy as np


keywords_apple = ["Apple", "Apple Inc", "iPhone", "iPod"]

data = pd.read_csv('news.csv')
#print(data.head())
data = data.dropna()
data = (data[data.keywords.str.contains('|'.join(keywords_apple))])
data["timestamp"] = data["timestamp"].apply(lambda date: date[:10])
#print(data.head())


#dateList = list(set(data['timestamp'].tolist()))
#dateList.sort()

#df = pd.read_csv('news.csv')

data['sentiment'] = ''

print(len(data))
#data = data.iloc[:, :len(data)/2]  #Talle
data = data.iloc[:, int(len(data)/2):] #David

print(data.head())

for index, row in data.iterrows():
    sentiment = -1
    while sentiment == -1:
        inp = input(row['headline'])
        if inp == "1":
            sentiment = "pos"
        elif inp == "2":
            sentiment = "neu"
        elif inp == "3":
            sentiment = "neg"
        else:
            sentiment = -1
    data.at[index, 'sentiment'] = sentiment


#data.to_csv('news_data_sent_talle.csv')
data.to_csv('news_data_sent_david.csv')