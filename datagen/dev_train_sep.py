import sqlite3 as lite
from random import shuffle
import sys, os

def help():
	print('''
------------------------------------------------------------------------------------------------------------------------
Creates tables for train and dev datasets

Usage: \tdev_train_sep.py [database file]

database file:\tdatabase file path.\t(default = plag.db)

Note: all paths must be relative to the parent folder of this script.
------------------------------------------------------------------------------------------------------------------------
	''')

def separate(c, c2, batch_size=1000, percent_dev=30):
	print('\nStarting train/dev tables insertion')

	c.execute('DROP TABLE IF EXISTS dataset_train')
	c.execute('DROP TABLE IF EXISTS dataset_dev')

	c.execute('CREATE TABLE dataset_train (id1 INTEGER NOT NULL, id2 INTEGER NOT NULL, same_style BOOL NOT NULL)')
	c.execute('CREATE TABLE dataset_dev (id1 INTEGER NOT NULL, id2 INTEGER NOT NULL, same_style BOOL NOT NULL)')

	insert_train = 'insert into dataset_train(id1, id2, same_style) values(?,?,?)'
	insert_dev = 'insert into dataset_dev(id1, id2, same_style) values(?,?,?)'

	c2.execute('SELECT * FROM dataset_id')
	l = c2.fetchmany(batch_size)
	l_size = len(l)
	i = 1
	while l_size > 0:
		shuffle(l)
		dev_idx = l_size * percent_dev // 100
		dev, train = l[:dev_idx], l[dev_idx:]

		c.executemany(insert_train, train)
		c.executemany(insert_dev, dev)
		print('Iteration {} complete ({} inserts per loop)'.format(i, batch_size))

		l = c2.fetchmany(batch_size)
		l_size = len(l)
		i = i + 1


if __name__ == '__main__':
	os.chdir('../')

	if len(sys.argv) > 2:
		help()
		raise ValueError('Invalid number of arguments. Received: ' + str(len(sys.argv)-1) + ' Expected: 1')
	if len(sys.argv) > 1:
		db_filename = sys.argv[1]
	else:
		db_filename = 'plag.db'

	try:
		db = lite.connect(db_filename)
		c = db.cursor()
		c2 = db.cursor()

		separate(c, c2)

		db.commit()
	except lite.Error as e:
		print("Error %s:" % e.args[0])
		sys.exit(1)
	finally:
		if db:
			db.close()
