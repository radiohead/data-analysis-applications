import math

import numpy as np

from matplotlib import pyplot
from multiprocessing import Process
from sklearn.neighbors import KNeighborsClassifier

def plot_histogram(data, sample_size):
  n, bins, _ = pyplot.hist(data, int(math.sqrt(len(data))), [min(data), max(data)], facecolor='green', alpha=0.75)
  pyplot.axis([min(bins), max(bins), min(n), max(n)])
  pyplot.grid(True)
  pyplot.xlim(0, 1)
  pyplot.xlabel('C-index')
  pyplot.ylabel('Samples')
  pyplot.title('Sample size = %d' % sample_size)
  pyplot.savefig('hist-%d-samples.png' % sample_size)
  pyplot.clf()

def generate_data(size):
  Y = np.ones(size)
  Y[:(size / 2)] = 0
  Y = np.random.permutation(Y)

  return (np.random.sample(size).reshape(size, 1), Y)

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

def process(sample_size):
  c_indices = []

  for i in xrange(0, 100):
    train, test = generate_data(sample_size)
    predicted = np.empty(test.shape[0])

    for i in xrange(0, train.shape[0]):
      train_t = np.delete(train, i).reshape(sample_size - 1, 1)
      test_t = np.delete(test, i)

      predicted[i] = predict(train_t, test_t, 3, train[i])

    c_indices.append(calculate_c_index(predicted, test))

  plot_histogram(c_indices, sample_size)

  fraction = reduce(lambda acc, succ: acc + 1 if succ > 0.7 else acc, c_indices, 0.0)

  print('Fraction for %d = %f' % (sample_size, fraction))
  print('Mean for %d = %f' % (sample_size, np.mean(c_indices)))
  print('Variance for %d = %f' % (sample_size, np.var(c_indices)))

if __name__ == '__main__':
  for sample_size in (10, 50, 100, 500):
    p = Process(target = process, args = (sample_size,))
    p.start()

  p.join()
