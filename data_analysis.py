import pandas as pd
from sklearn.utils import shuffle
import numpy as np

def pos_neg_neu(x):
    if x == 'positive':
        return 1
    elif x == 'negative':
        return -1
    else:
        return 0

data = pd.read_csv('data/news_development_to_emotions_relative_morefeatures.csv')
print(data.columns)

#split data
data_neg1 = data[data['development'] == -1]
data_0 = data[data['development'] == 0]
data_1 = data[data['development'] == 1]

#find correlations
features = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust', 'sentiment_dict', 'sent_compound','sent_neg','sent_neu','sent_pos' ]
print("development:     -1         |       0         |        1")
for feature in features:
    neg = data_neg1[feature].mean()
    neu = data_0[feature].mean()
    pos = data_1[feature].mean()
    print("{}: {} | {} | {}".format(feature, neg, neu, pos))