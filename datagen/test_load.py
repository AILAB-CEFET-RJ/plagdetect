import sqlite3 as lite
from generate_dataset import get_id_dataset, get_sentences_hashmap

#edited
db = lite.connect('../plag.db')
c = db.cursor()
hashmap = get_sentences_hashmap(c)
get_id_dataset(c)
tuples = c.execute().fetchall()
