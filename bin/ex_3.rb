# normalize using z-score
# leave-out-out and calc C-index
require 'rubygems'
require 'bundler/setup'
require 'knn'
require 'csv'

def process
  csv = CSV.open('data/Water_data.csv', headers: true)
  data = csv.map { |row| row.map { |_, v| v.to_f } }

  KNN::Normalizations::ZScore.new(data).normalize([3, 4, 5])
end

def run(data, strategy)
  (1..5).map do |k|
    fork do
      puts "K: #{k} - #{strategy.new(data, [0, 1, 2], k).regression.inspect}"
    end
  end

  Process.waitall
end

data = process

puts 'One out'
run(data, Validations::LeaveOneOutMulti)

puts 'Three out'
run(data, Validations::LeaveThreeOutMulti)
