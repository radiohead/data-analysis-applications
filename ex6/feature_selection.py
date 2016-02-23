import operator
import numpy as np

from scipy.stats import pearsonr
from multiprocessing import Process
from sklearn.neighbors import KNeighborsClassifier

def generate_data(size):
  Y = np.ones(size)
  Y[:(size / 2)] = 0
  Y = np.random.permutation(Y)

  return (np.random.sample((size, 1000)), Y)

def select_features(X, Y, size):
  features = {}

  for i in xrange(0, len(X)):
    features[i] = abs(pearsonr(X[:,i], Y)[0])

  sorted_features = sorted(features.items(), key=operator.itemgetter(1))
  return [x[0] for x in sorted_features[len(sorted_features) - size:len(sorted_features)]]

def predict(X, Y, N, row):
  return KNeighborsClassifier(n_neighbors=N).fit(X, Y).predict([row])

def calculate_c_index(predicted, actual):
  n = 0.0
  h_sum = 0.0
  actual_len = len(actual)

  for i in xrange(0, actual_len):
    t = actual[i]
    p = predicted[i]

    for j in xrange(i + 1, actual_len):
      nt = actual[j]
      np = predicted[j]

      if t != nt:
        n = n + 1.0

        if ((p < np) and (t < nt)) or ((p > np) and (t > nt)):
          h_sum = h_sum + 1.0
        elif ((p < np) and (t > nt)) or ((p > np) and (t < nt)):
          pass
        elif p == np:
          h_sum = h_sum + 0.5

  return h_sum / n

def process_wrong():
  train, test = generate_data(50)
  features = select_features(train, test, 10)

  predicted = np.empty(test.shape[0])
  train = train[:, features]

  for i in xrange(0, train.shape[0]):
    train_t = np.delete(train, (i), axis=0)
    test_t = np.delete(test, i)

    predicted[i] = predict(train_t, test_t, 3, train[i])

  print 'C-index for wrong = %f' % calculate_c_index(predicted, test)

def process_right():
  train, test = generate_data(50)
  predicted = np.empty(test.shape[0])

  for i in xrange(0, train.shape[0]):
    train_t = np.delete(train, (i), axis=0)
    test_t = np.delete(test, i)

    features = select_features(train_t, test_t, 10)
    train_t = train_t[:, features]

    predicted[i] = predict(train_t, test_t, 3, train[i, features])

  print 'C-index for right = %f' % calculate_c_index(predicted, test)

if __name__ == '__main__':
  process_wrong()
  process_right()
