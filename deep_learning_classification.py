import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Convolution2D


counter = 0

def drop_it_like_its_hot(x):
    global counter
    if counter < 13000 and x == 0:
            counter += 1
            return -1
    else:
        return x


def pos_neg_neu(x):
    if x == 'positive':
        return 1
    elif x == 'negative':
        return -1
    else:
        return 0

data = pd.read_csv('news_to_emotions.csv')
data = data.drop(['market_date', 'company_symbol'], axis=1)
data['sentiment'] = data['sentiment'].apply(lambda x: pos_neg_neu(x))
data['is_volatile'] = data['is_volatile'].apply(lambda x: drop_it_like_its_hot(x))
data = data[~(data['is_volatile'] == -1.0)]

x_train, x_test, y_train, y_test = train_test_split(data.drop(['is_volatile'], axis=1), data['is_volatile'], test_size=0.2)

model = Sequential()
model.add



