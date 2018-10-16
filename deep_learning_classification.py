from keras.models import Sequential
from keras.layers import Dense
import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn.utils import shuffle

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
data = shuffle(data)

x_train, x_test, y_train, y_test = train_test_split(data.drop(['is_volatile'], axis=1), data['is_volatile'], test_size=0.2)
x_train, x_test, y_train, y_test = x_train.values, x_test.values, y_train.values, y_test.values

x_train_val = x_train[:int(len(x_train)*0.8)]
x_val = x_train[int(len(x_train)*0.8):]
y_train_val = y_train[:int(len(y_train)*0.8)]
y_val =y_train[int(len(y_train)*0.8):]

model = Sequential()
model.add(Dense(units=64, activation='relu', input_shape=(9,)))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(x_train_val, y_train_val, epochs=100, batch_size=32, verbose=0)
modelname = "model_{}".format(1)
model.save('models/{}.h5'.format(modelname))

loss_and_metrics = model.evaluate(x_val, y_val, batch_size=128)
with open("loss_and_accuracy.txt", "a") as file:
    file.write("{}: {}\n".format(modelname, loss_and_metrics))
