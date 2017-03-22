
from imdb import IMDb

ia = IMDb()

#mov = ia.search_episode('Lost')


mov = ia.search_movie('Braveheart')

first = mov[0]

firstID = ia.get_movie(first.movieID)
print(firstID)

for elem in  firstID['cast']:
    print elem
'''
if 'Wallace' in firstID['cast'][0]:
    print "in cast"
else:
    print "not here"

print(first['kind'])
#print(first['genre'])
'''


#movID = ia.title2imdbID('Braveheart')


#print(mov)

#attempt = ia.get_movie('0112573')
#print(attempt)
#print(attempt['cast'])
#print(attempt['characters'])

#print(mov[0]['cast'])









#print(mov)
#print(mov[4])

#print(mov[0].summary())

#ia.update(mov)

#print(mov[0].summary())

#print(mov[0].characters())

#print(mov[0])

#first = mov[0]

#print(first.summary())










'''
import imdb

ia = imdb.IMDb()

s_result = ia.search_movie('YoungDracula')

print s_result[0]['director']
for item in s_result:
   print item


'''


