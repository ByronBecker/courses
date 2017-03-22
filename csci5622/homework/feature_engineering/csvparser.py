import csv


f = open('toytrain.csv', 'r+')

csv_f = csv.reader(f)
row = f.next()
row.append(8)
print row




