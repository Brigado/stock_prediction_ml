import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

ps = PorterStemmer()

emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust', 'positive', 'negative']
col = ['market_date', 'company_symbol', 'sentiment', 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness','surprise', 'trust', 'is_volatile']
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
                emotion_score[emotion] += 1

    #get sentiment
    if emotion_score["positive"] == emotion_score["negative"]:
        sentiment = "neutral"
    else:
        sentiment = "positive" if emotion_score["positive"] > emotion_score["negative"] else "negative"

    return emotion_score, sentiment

#dataset:
df = pd.read_csv('data/news_to_volatility_dataset.csv')
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
        df_result.to_csv("news_to_emotions{}.csv".format(i))

    for company in companyList:
        df_date_company = df.loc[(df['market_date'] == date) & (df['company_symbol'] == company), 'headline'].values
        # create bag of stemmed words out of headlines
        wordset = set()
        for hl in df_date_company:
            words = word_tokenize(hl)
            for word in words:
                wordset.add(ps.stem(word))
        emotions, sentiment_str = get_emotions(wordset)

        if len(wordset) > 0:
            volatility = df.loc[(df['market_date'] == date) & (df['company_symbol'] == company), 'is_volatile'].values[0]
            lst.append([date, company, emotions['anger'], emotions['anticipation'], emotions['disgust'], emotions['fear'], emotions['joy'], emotions['sadness'], emotions['surprise'], emotions['trust'], sentiment_str, volatility])

df_result = pd.DataFrame(lst, columns=col)
df_result.to_csv("news_to_emotions.csv")