import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

ps = PorterStemmer()

emotions = ['positive', 'negative', 'litigious', 'uncertainty', 'constraining', 'strong_modal', 'moderate_modal', 'weak_modal']
col = ['market_date', 'company_symbol', 'positive', 'negative', 'litigious', 'uncertainty', 'constraining', 'strong_modal', 'moderate_modal', 'weak_modal', 'sentiment_lib_avg', 'is_volatile']
emotion_dfs = {}
dict_path_dir = "data/dictionaries/"
for dict in os.listdir(dict_path_dir):
    dict_name = dict[:-4]
    emotion_dfs.update({dict_name: (pd.read_csv(dict_path_dir + dict))})

def get_emotions(wordset):
    #search for words in dictionaries
    emotion_score = {}
    for emotion in emotions:
        emotion_score[emotion] = 0
    for emotion in emotion_dfs:
        df = emotion_dfs[emotion]
        for word in wordset:
            if word in df['word'].values:
                #print(word)
                weight = df.loc[df['word'] == word, ['weight']].values[0]
                #print(type(weight[0]))
                #print(weight)
                #print('\n')
                emotion_score[emotion] += weight[0]

    return emotion_score

#dataset:
df = pd.read_csv('data/stock_data/news_to_volatility_dataset.csv')
lst = []

#for all headline for one market day:
dateList = list(set(df['market_date'].tolist()))
companyList = list(set(df['company_symbol'].tolist()))
dateList.sort()
i = 0
for date in dateList:
    i+=1
    if (i%500 == 0):
        print(i)
        df_result = pd.DataFrame(lst, columns=col)
        df_result.to_csv("news_to_emotions_{}.csv".format(i))

    for company in companyList:
        df_date_company = df.loc[(df['market_date'] == date) & (df['company_symbol'] == company), 'headline'].values
        # create bag of stemmed words out of headlines
        wordset = set()
        sentiments = []
        for hl in df_date_company:
            #compute sentiment with library:
            sia = SIA()
            pol_score = sia.polarity_scores(hl)
            sentiments.append(pol_score['compound'])

            #dictionary preparation
            words = word_tokenize(hl)
            for word in words:
                wordset.add(word.lower())
        emotions = get_emotions(wordset)

        if len(wordset) > 0:
            volatility = df.loc[(df['market_date'] == date) & (df['company_symbol'] == company), 'is_volatile'].values[0]
            lst.append([date, company, float(emotions['positive']), float(emotions['negative']), float(emotions['litigious']), float(emotions['uncertainty']), float(emotions['constraining']), float(emotions['strong_modal']), float(emotions['moderate_modal']), float(emotions['weak_modal']), sum(sentiments), volatility])

df_result = pd.DataFrame(lst, columns=col)
df_result.to_csv("news_to_emotions.csv")