require 'rubygems'
require 'bundler/setup'
require 'knn'
require 'csv'

def run(file)
  puts 'Starting'

  csv = CSV.open(file)
  data = csv.map { |row| row.map{ |c| c.to_f } }

  (5..10).map do |number_of_neighbors|
    fork do
      target_index = data.first.size - 1

      strategy = Validations::KFoldConfusionClassification.new(data, target_index, number_of_neighbors, 10)
      puts "K: #{number_of_neighbors} - #{strategy.classification}"
    end
  end

  Process.waitall
end

def run_test(file, k)
  puts 'Starting'

  csv = CSV.open(file)
  data = csv.map { |row| row.map{ |c| c.to_f } }

  target_index = data.first.size - 1

  strategy = Validations::KFoldConfusionClassification.new(data, target_index, k, 10)
  puts "#{strategy.classification}"
  puts "#{strategy.confusion_matrix}"
end

run('data/TrainData.csv')
run_test('data/TestData.csv', 5)
