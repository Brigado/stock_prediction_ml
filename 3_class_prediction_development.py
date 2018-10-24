import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras import metrics

counter = 0
def drop_it_like_its_hot(x):
    global counter
    if counter < 16750 and x == 0:
            counter += 1
            return -2
    else:
        return x

def evaluate(x_test, y_test, model):
    neg_right, neu_right, pos_right, neg_count, neu_count, pos_count = 0,0,0,0,0,0
    for idx, x in enumerate(x_test.values):
        pred = model.predict(np.array([x,]))
        if y_test[idx][0] == 1:#neg
            neg_count+=1
            if np.argmax(pred[0]) == 0:
                neg_right += 1
        elif y_test[idx][1] == 1:#neu
            neu_count += 1
            if np.argmax(pred[0]) == 1:
                neu_right += 1
        else:#pos
            pos_count += 1
            if np.argmax(pred[0]) == 2:
                pos_right += 1

    print("neg: {} from {}: {} % \nneu: {} from {}: {} % \npos: {} from {}: {} % \n".format(neg_right, neg_count, 100*float(neg_right)/float(neg_count), neu_right, neu_count, 100*float(neu_right)/float(neu_count), pos_right, pos_count, 100*float(pos_right)/float(pos_count)))
    #print("neg: {} from {}: {} % \npos: {} from {}: {} % \n".format(neg_right, neg_count, 100*float(neg_right)/float(neg_count), pos_right, pos_count, 100*float(pos_right)/float(pos_count)))

data = pd.read_csv('data/news_development_to_emotions_relative_morefeatures.csv')
data = data.drop([data.columns[0], 'market_date', 'company_symbol'], axis=1)
#data = data.drop(['anger','anticipation','disgust','fear','joy','sadness','surprise','trust','sentiment_dict'], axis =1)
data['development'] = data['development'].apply(lambda x: drop_it_like_its_hot(x))
data = data[~(data['development'] == -2)]
data = shuffle(data)
x_train, x_test, y_train, y_test = train_test_split(data.drop(['development'], axis=1), data['development'], test_size=0.2)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y_train)
encoder.fit(y_test)

encoded_Y = encoder.transform(y_train)
encoded_Y_test = encoder.transform(y_test)

# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
dummy_y_test = np_utils.to_categorical(encoded_Y_test)

model = Sequential()
model.add(Dense(64, input_dim=13, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', metrics.categorical_accuracy])

#estimator = KerasClassifier(build_fn=model_fun, epochs=1, batch_size=5, verbose=0)
print("training:")

# kfold = KFold(n_splits=10, shuffle=True, random_state=42)
# results = cross_val_score(estimator, x_train, dummy_y, cv=kfold)
# print(results)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

#pred = estimator.predict(y_train)

model.fit(x_train, dummy_y, epochs=150, batch_size=64, verbose=0)
loss_and_metrics = model.evaluate(x_test, dummy_y_test, batch_size=64)
print(loss_and_metrics)
evaluate(x_train, dummy_y, model)
evaluate(x_test, dummy_y_test, model)