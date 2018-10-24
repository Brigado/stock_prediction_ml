import pandas as pd
import os
from nltk.tokenize import sent_tokenize, word_tokenize

dict_path = 'data/dictionaries/'
words = []
most_common = 0

def compute_weight(word):
    global words, most_common
    count = words.count(word)
    if count == 0:
        return 0
    else:
        #print(word)
        #print(count)
        #print(most_common)
        #print('\n\n')
        return 1 - 0.5 * (count/most_common)


vol = pd.read_csv('data/stock_data/news_to_volatility_dataset.csv')
for index, row in vol.iterrows():
    if index % 100 == 0 and index != 0:
        print(index)
    hl = row['headline']
    tokens = word_tokenize(str(hl))
    for token in tokens:
        words.append(str(token).lower())
most_common = words.count(max(set(words), key=words.count))
print(most_common)

for dict in os.listdir(dict_path):
    df = pd.read_csv(dict_path + dict)
    print(dict)
    for index, row in df.iterrows():
        if index % 100 == 0:
            print(index)
        weight = compute_weight(str(row['word']).lower())
        df.loc[index, 'weight'] = weight
    df = df[~(df['weight'] == 0)]
    df['word'] = df['word'].apply(lambda x: str(x).lower())
    df.to_csv(dict_path + 'modified_' + dict, index=None)
