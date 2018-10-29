import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

emotion_values = ['positive', 'negative', 'litigious', 'uncertainty', 'constraining', 'strong_modal', 'moderate_modal', 'weak_modal']
emotion_dfs = {}
dict_path_dir = "data/dictionaries/"
for dict in os.listdir(dict_path_dir):
    dict_name = dict[:-4]
    emotion_dfs.update({dict_name: (pd.read_csv(dict_path_dir + dict))})

def get_emotions(wordset):
    #search for words in dictionaries
    emotion_score = {}
    for emotion in emotion_values:
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
grouped = False
for relative in (True, False):
    print("relative: " + str(relative))
    r = "_relative" if relative else ""
    for grouped in (True, False):
        print("grouped: " + str(grouped))
        df = None
        name = ""
        if grouped:
            df = pd.read_csv('data/stock_data/balanced_grouped_dataset.csv')
            print("count 0: " + str(df[df['is_volatile']==0].count()['is_volatile']))
            print("count 1: " + str(df[df['is_volatile'] == 1].count()['is_volatile']))
            name = "balanced_grouped_dataset_sentiment{}.csv".format(r)
            col = ['market_date', 'company_symbol', 'headline', 'positive', 'negative', 'litigious', 'uncertainty', 'constraining','strong_modal', 'moderate_modal', 'weak_modal', 'sentiment_lib_avg', 'is_volatile']


        else:
            df = pd.read_csv('data/stock_data/balanced_dataset.csv')
            print("count 0: " + str(df[df['is_volatile']==0].count()['is_volatile']))
            print("count 1: " + str(df[df['is_volatile'] == 1].count()['is_volatile']))
            name = "balanced_dataset_sentiment{}.csv".format(r)
            col = ['market_date', 'company_symbol', 'news_id', 'headline', 'positive', 'negative', 'litigious', 'uncertainty', 'constraining','strong_modal', 'moderate_modal', 'weak_modal', 'sentiment_lib_avg', 'is_volatile']

        lst = []
        for idx, row in df.iterrows():
            if (idx+1) % 1000 == 0:
                print(idx)
            sentiment = 0
            if grouped:
                headlines = str(row['headline']).split("<s>")
                sentiments = []
                for hl in headlines:
                    sia = SIA()
                    pol_score = sia.polarity_scores(hl)
                    sentiments.append(pol_score['compound'])
                sentiment = sum(sentiments)/float(len(sentiments))

            else:
                hl = row['headline']
                # compute sentiment with library:
                sia = SIA()
                sentiment = sia.polarity_scores(hl)['compound']

            # dictionary preparation
            words = word_tokenize(row['headline'])
            wordset = []
            for word in words:
                wordset.append(word.lower())
            emotions = get_emotions(wordset)

            c = float(len(wordset)) if relative else 1
            if grouped:
                lst.append([row['market_date'], row['company_symbol'], row['headline'], float(emotions['positive'])/c,
                            float(emotions['negative'])/c, float(emotions['litigious'])/c,
                            float(emotions['uncertainty'])/c, float(emotions['constraining'])/c,
                            float(emotions['strong_modal'])/c,float(emotions['moderate_modal'])/c,
                            float(emotions['weak_modal'])/c, sentiment, row['is_volatile']])
            else:
                lst.append([row['market_date'], row['company_symbol'], row['news_id'], row['headline'], float(emotions['positive']) / c,
                            float(emotions['negative']) / c, float(emotions['litigious']) / c,
                            float(emotions['uncertainty']) / c, float(emotions['constraining']) / c,
                            float(emotions['strong_modal']) / c, float(emotions['moderate_modal']) / c,
                            float(emotions['weak_modal']) / c, sentiment, row['is_volatile']])

        df_result = pd.DataFrame(lst, columns=col)
        df_result.to_csv(name)