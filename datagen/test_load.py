import sqlite3 as lite
from generate_dataset import get_id_dataset, get_sentences_hashmap, get_id_dataset_size

#edited
db = lite.connect('../plag.db')
c = db.cursor()

print 'Loading hashmap'
hashmap = get_sentences_hashmap(c)
print 'Hashmap loaded'

print 'Executing count SQL'
size = get_id_dataset(c)
print 'Number of tuples'
print size

print 'SQL executed'
get_id_dataset(c)
tuples = c.fetchall()
print 'Tuples loaded in memory'
raw_input("Test completed. Press Enter to continue...")

