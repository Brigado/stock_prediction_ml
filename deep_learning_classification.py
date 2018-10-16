from keras.models import Sequential
from keras.layers import Dense
import keras
import pandas as pd
from sklearn.model_selection import train_test_split

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
data = data.drop(data.columns[0], axis=1)
data = data.drop(['market_date', 'company_symbol'], axis=1)
data['sentiment'] = data['sentiment'].apply(lambda x: pos_neg_neu(x))
data['is_volatile'] = data['is_volatile'].apply(lambda x: drop_it_like_its_hot(x))
data = data[~(data['is_volatile'] == -1.0)]
data = data[12000:]

x_train, x_test, y_train, y_test = train_test_split(data.drop(['is_volatile'], axis=1), data['is_volatile'], test_size=0.2)
x_train, x_test, y_train, y_test = x_train.values, x_test.values, y_train.values, y_test.values

model = Sequential()
model.add(Dense(units=64, activation='relu', input_shape=(9,)))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=32)

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
