require 'rubygems'
require 'bundler/setup'
require 'knn'
require 'csv'

def process(filename)
  csv = CSV.open(filename)
  data = csv.map { |row| row.map(&:to_f) }

  normalized = KNN::Normalizations::ZScore.new(data).normalize((0...data.first.size))

  name, ext = filename.split('.')
  name = "#{name}_normalized"

  CSV.open("#{name}.#{ext}", 'wb') do |out|
    normalized.each do |row|
      out << row
    end
  end
end

%w(data/INPUT.csv data/OUTPUT.csv).each do |f|
  process(f)
end
