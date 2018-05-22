import sqlite3 as lite
import generate_dataset as gd


if __name__ == '__main__':
	db = lite.connect('../plag.db')
	c = db.cursor()
	gd.get_author_ids(c)
	authors = c.fetchall()
	gd.get_dataset_by_author_id(c, authors[0])
	a = c.fetchall()
	print len(a)
	gd.get_dataset(c)
	a = c.fetchmany(1024)
	print(a)
