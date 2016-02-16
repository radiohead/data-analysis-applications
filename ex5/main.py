import numpy as np

from multiprocessing import Process
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import euclidean

def filter_data(input, output, x, y, comparator):
  training = []
  result = []

  size_x = input.shape[0]
  size_y = input.shape[1]

  for a in xrange(0, size_x):
    for b in xrange(0, size_y):
      if comparator(x, y, a, b):
        training.append(input[a][b])
        result.append(output[a * size_x + b])

  return (np.array(training), np.array(result))

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

def process(input_set, output_set, N, comparator):
  predicted = np.empty(output_set.shape[0])
  size_x = input_set.shape[0]
  size_y = input_set.shape[1]

  for x in xrange(0, size_x):
    for y in xrange(0, size_y):
      train, test = filter_data(input_set, output_set, x, y, comparator)
      predicted[x * size_x + y] = predict(train, test, N, input_set[x][y])

  c_index = calculate_c_index(predicted, output_set)

  print 'C is ' + str(c_index)

if __name__ == '__main__':
  input = np.genfromtxt('proteins.features', delimiter=',')
  input = input.reshape((20, 20, input.shape[-1]))

  output = np.genfromtxt('proteins.labels', delimiter=',')

  for N in xrange(1, 20):
    print 'Non-modified for N = ' + str(N)
    process(input, output,  N, lambda x, y, a, b: x != a or y != b)

  for N in xrange(1, 20):
    print 'Modified for N = ' + str(N)
    process(input, output, N, lambda x, y, a, b: x != a and x != b and y != a and y != b)
