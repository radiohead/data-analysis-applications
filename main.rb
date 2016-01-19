require 'rubygems'
require 'bundler/setup'
require 'knn'
require 'csv'

def run(data, type)
  means_and_vars = (2..15).map do |number_of_neighbors|
    target_index = data.first.size - 1

    leave_one_out = Validations::LeaveOneOut.new(data, target_index, number_of_neighbors)
    number_of_neighbors, mean_error, error_variance = leave_one_out.send(type)

    puts "#{type.capitalize}: Mean error for K = #{number_of_neighbors} is #{mean_error}"
    puts "#{type.capitalize}: Variance for   K = #{number_of_neighbors} is #{error_variance}"

    [number_of_neighbors, mean_error, error_variance]
  end

  puts "#{type.capitalize}: Least mean error is #{means_and_vars.min_by { |vals| vals[1] }}"
  puts "#{type.capitalize}: Least variance is #{means_and_vars.min_by { |vals| vals[2] }}"
  puts "#{type.capitalize}: Best overall is #{means_and_vars.min_by { |vals| (vals[1] + vals[2]) / 2 }}"  
end

csv = CSV.open('data/processed.cleveland.data')
data = csv.map{ |row| row.map{ |c| c.to_f rescue 0.0 } }
run(data, :regression)
run(data, :classification)
