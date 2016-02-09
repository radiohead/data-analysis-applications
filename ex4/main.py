import numpy as np

from multiprocessing import Process
from sklearn.neighbors import KNeighborsRegressor
from scipy.spatial.distance import euclidean

def filter_data(input, output, distance, pivot_distance, threshold):
  training = []
  result = []

  for i in xrange(0, len(input)):
    if euclidean(pivot_distance, distance[i]) > threshold:
      training.append(input[i])
      result.append(output[i])

  return [np.array(training), np.array(result)]

def predict(X, Y, row):
  return KNeighborsRegressor(n_neighbors=5).fit(X, Y).predict([row])

def calculate_c_index(predicted, actual):
  n = 0.0
  h_sum = 0.0
  actual_len = len(actual)

  for i in xrange(0, actual_len):
    # print 'C-index: ' + str(i) + ' out of ' + str(actual_len)
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

def process(input_set, output_set, distance_set, threshold):
  predicted = np.empty(len(output_set))
  size = len(input_set)

  for i in xrange(0, size):
    # print 'Going over ' + str(i) + ' out of ' + str(size)
    filtered = filter_data(input_set, output_set, distance_set, distance[i], threshold)
    predicted[i] = predict(filtered[0], filtered[1], input_set[i])

  c_index = calculate_c_index(predicted, output_set)

  print 'C for T = ' + str(threshold) + ' is ' + str(c_index)

if __name__ == '__main__':
  input = np.genfromtxt('INPUT.csv', delimiter=',')
  output = np.genfromtxt('OUTPUT.csv', delimiter=',')
  distance = np.genfromtxt('COORDINATES.csv', delimiter=',')

  for t in xrange(0, 210, 10):
    p = Process(target = process, args = (input, output, distance, t))
    p.start()

  p.join()
