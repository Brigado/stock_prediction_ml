import pandas as pd
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense

data = pd.read_csv('data/news_development_to_emotions_relative_morefeatures.csv')
data = data.drop([data.columns[0], 'market_date', 'company_symbol'], axis=1)
data = data[~(data['development'] == 0)]
data = shuffle(data)
#print(data.development.value_counts())

x_train, x_test, y_train, y_test = train_test_split(data.drop(['development'], axis=1), data['development'], test_size=0.2)

def knn_prediction():
    knn = neighbors.KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train, y_train)
    print("knn: {}".format(knn.score(x_test, y_test)))

def naive_bayes():
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    print("clf: {}".format(clf.score(x_test, y_test)))

def nn():
    global x_train, x_test, y_train, y_test
    x_train_, x_test_, y_train_, y_test_ = x_train.values, x_test.values, y_train.values, y_test.values

    model = Sequential()
    model.add(Dense(units=64, activation='tanh', input_shape=(13,)))
    model.add(Dense(units=128, activation='tanh'))
    model.add(Dense(units=128, activation='tanh'))
    model.add(Dense(units=1, activation='tanh'))

    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    model.fit(x_train_, y_train_, epochs=300, batch_size=64, verbose=0)
    loss_and_metrics = model.evaluate(x_test_, y_test_, batch_size=64)
    print(loss_and_metrics)

knn_prediction()
naive_bayes()
nn()