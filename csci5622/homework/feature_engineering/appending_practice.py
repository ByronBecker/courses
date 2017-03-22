
import csv


'''
with open('prac_test.csv', 'w') as w:
    for i in range(0,100):
        w.write('Hello,World\n')

'''
'''
with open('prac_test.csv', 'r') as csvin:
    with open('prac_test.csv', 'a') as csvout:
        #wr = csv.writer(csvout, lineterminator='\n')
        rd = csv.reader(csvin)

        for row in csvin:
            #row = row + w.write('hello')
            row = csvout.write(row + ',hello')
            print row
'''
'''
with open('prac_test.csv', 'ra+') as f:
    #reader = csv.reader(f, delimiter=',')
    wr = csv.writer(f)
    for row in f:
        row = row.split('\n')
        wr.writerow(['-'.join(row)])
        print row
        row.replace('\n', 'hello\n')
        print row
        first = row.split('\n')
        print first
        first[1] = ',hello'
        print first
        print row
        '''

with open('prac_test.csv', 'r+') as f:
    reader = csv.reader(f)

    rows = []
    for row in reader:
        row.replace('\n', ',butter\n')
        print row
        rows.append(row)
    '''
    writer = csv.writer(f)
    writer.writerows(rows)
    '''

    '''
    print(lines)
    f = open('prac_test.csv', 'w')
    for line in lines:
        line.replace('\n', 'butter')
        

        f.write(''.join(lines))

    f.close()
'''
'''
with open('prac_test.csv', 'w') as myread:
    for line in myread:
        myread.write(",butter\n")

#wr = csv.writer(w,delimiter=',')
for row in w
    print row
    row.append('its me')
#for i in range(0, 100):
   # wr.writerow(['Hello','World\n'])


for line in w:
    wr.writerow(line + ',butter')
w.close()
'''
