
# coding: utf-8

# # Logistic Regression in Python

# ## Load the iris dataset

# In[1]:

import pandas as pd
df_iris = pd.read_csv('iris.csv', header=None)
df_iris.columns = ['sepal_length', 'sepal_width', 'label_str']

# preprocessing
df_iris['bias'] = 1
str_to_int = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
df_iris['label_int'] = df_iris['label_str'].apply(lambda label_str: str_to_int[label_str])
df_iris.tail()


# In[2]:

# plot the iris dataset

import matplotlib.pyplot as plt
import numpy as np

# plot data

plt.scatter(df_iris.iloc[:50, 0], df_iris.iloc[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(df_iris.iloc[50:100, 0], df_iris.iloc[50:100, 1],
            color='blue', marker='x', label='versicolor')
plt.scatter(df_iris.iloc[100:150, 0], df_iris.iloc[50:100, 1],
            color='green', marker='^', label='virginica')

plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()


# In[3]:

from sklearn.model_selection import train_test_split

# select data, features, labels
X = df_iris[['sepal_length', 'sepal_width', 'bias']]
y = df_iris['label_int']
X, y = X[:100], y[:100] # use 2 labels only (binary classification)

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# ## Part 1: Sklearn Implementation
# ### Adapted from Justin Markham's Jupyter Notebook

# In[4]:

from sklearn.linear_model import LogisticRegression

# instantiate and fit model
model_1 = LogisticRegression()
model_1 = model_1.fit(X_train, y_train)

# evaluate model
train_acc =  model_1.score(X_train, y_train)
test_acc =  model_1.score(X_test, y_test)
print 'Train accuracy: {}'.format(train_acc)
print 'Test accuracy: {}'.format(test_acc)


# ## Part 2: Tflearn Implementation
# ###  Adapted from Tensorflow Documentation

# In[5]:

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)


# define model input
def input_fn(X_train, y_train):
    feature_cols = {name: tf.constant(X_train[name].values) for name in ['sepal_length', 'sepal_width']}
    label = tf.constant(y_train.values)
    return feature_cols, label


# define features
sepal_length = tf.contrib.layers.real_valued_column('sepal_length')
sepal_width = tf.contrib.layers.real_valued_column('sepal_width')

# instantiate and fit model
model_2 = tf.contrib.learn.LinearClassifier(feature_columns=[sepal_length, sepal_width])
model_2.fit(input_fn=lambda: input_fn(X_train, y_train), steps=100)

# evaluate model
train_acc = model_2.evaluate(input_fn=lambda: input_fn(X_train, y_train), steps=1)['accuracy']
test_acc = model_2.evaluate(input_fn=lambda: input_fn(X_test, y_test), steps=1)['accuracy']
print 'Train accuracy: {}'.format(train_acc)
print 'Test accuracy: {}'.format(test_acc)


# ## Part 3: Pure Python Implementation
# ###  From Sebastian Raschka's Github Repository

# In[6]:

class LogisticRegressorPurePython(object):
    
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            
            cost = -y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output)))
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.activation(X) >= 0.5, 1, 0)
    
    def activation(self, X):
        z = self.net_input(X)
        sigmoid = 1.0 / (1.0 + np.exp(-z))
        return sigmoid
    

# instantiate and fit model
model_3 = LogisticRegressorPurePython(n_iter=100, eta=0.001)
model_3.fit(X_train, y_train)

# evaluate model
train_acc = np.sum(model_3.predict(X_train) == y_train) / float(len(X_train))
test_acc = np.sum(model_3.predict(X_test) == y_test) / float(len(X_test))
print 'Train accuracy: {}'.format(train_acc)
print 'Test accuracy: {}'.format(test_acc)

