from collections import defaultdict
import numpy as np
food_list = 'spam spam spam spam spam spam eggs spam'.split()
food_count = defaultdict(int) # default value of int is 0
for food in food_list:
    food_count[food] += 1 # increment element's value by 1

#defaultdict(<type 'int'>, {'eggs': 1, 'spam': 7})
print food_count






values = np.array([3,2,1,4,0])

d = defaultdict(int)

d[1] = values

print d[1]

print d[0]
