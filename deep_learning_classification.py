import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data

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


def train(model, trainloader, validationloader, lossfunction, optimizer, n_epochs=100):
    trainingLosses, validationLosses = [], []
    for t in range(n_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels).float()  # See the comments below (1)

            optimizer.zero_grad()  # See the comments below (2)

            outputs = model(inputs)  # See the comments below (3)

            loss = lossfunction(outputs, labels)  # Compute the loss
            loss.backward()  # Compute the gradient for each variable
            optimizer.step()  # Update the weights according to the computed gradient

            # for printing
            running_loss += loss.data[0]

        # This second loop is actually just calculating the loss in the validation set
        # Otherwise, it's the same as above
        running_loss_val = 0.0
        for i, data in enumerate(validationloader):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels).float()
            outputs = model(inputs)
            loss = lossfunction(outputs, labels)  # Compute the loss

            # for printing
            running_loss_val += loss.data[0]
        trainingLosses.append(running_loss)
        validationLosses.append(running_loss_val)
        print("Epoch: {} Training loss: {:f} Validation loss: {:f}".format(t + 1, running_loss, running_loss_val))
    return trainingLosses, validationLosses


data = pd.read_csv('news_to_emotions.csv')
data = data.drop(['market_date', 'company_symbol'], axis=1)
data['sentiment'] = data['sentiment'].apply(lambda x: pos_neg_neu(x))
data['is_volatile'] = data['is_volatile'].apply(lambda x: drop_it_like_its_hot(x))
data = data[~(data['is_volatile'] == -1.0)]

x_train, x_test, y_train, y_test = train_test_split(data.drop(['is_volatile'], axis=1), data['is_volatile'], test_size=0.2)
x_train, x_test, y_train, y_test = torch.from_numpy(x_train.values), torch.from_numpy(x_test.values),torch.from_numpy(y_train.values),torch.from_numpy(y_test.values)


trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train.float(), y_train.float().view(-1,1)), batch_size=100)
testloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test.float(), y_test.float().view(-1,1)), batch_size=100)

net = nn.Sequential(
    nn.Linear(9, 1),
    nn.Sigmoid()
)

loss = nn.BCELoss()
opt = optim.SGD(net.parameters(), 1e-1)



