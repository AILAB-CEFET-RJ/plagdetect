import sqlite3 as lite
from generate_dataset import get_id_dataset, get_sentences_hashmap

#edited
db = lite.connect('../plag.db')
c = db.cursor()
hashmap = get_sentences_hashmap(c)
print 'Hashmap loaded in memory'
get_id_dataset(c)
print 'SQL executed'
tuples = c.fetchall()
print 'Tuples loaded in memory'
raw_input("Test completed. Press Enter to continue...")

