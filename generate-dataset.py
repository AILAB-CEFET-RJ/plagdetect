import sqlite3 as lite
import os, sys, errno

def create_folder(directory):
	try:
		os.makedirs(directory)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise

def generate_file(c2, c3, doc_id):
	c2.execute('SELECT id, fragment, isplag FROM sentence where fk_article_id = ? ORDER BY offset', (doc_id,))
	#if c2.rowcount > 0:
	with open(os.path.join(directory, 'suspicious-document' + f'{doc_id:05}' + '.txt'), 'w') as f:
		for row in c2:
			c3.execute('SELECT fragment, isplag FROM sentence where fk_article_id = ? AND id > ? ORDER BY offset', (doc_id, row[0]))
			for row_ahead in c3:
				same_style = '1' if row[2] == row_ahead[1] else '0'
				f.write(row[1] + '\t' + row_ahead[0] + '\t' + same_style + '\n')


if __name__ == '__main__':
	global directory
	directory = 'dataset-generated'
	create_folder(directory)

	db_filename = 'plag.db'
	db = None
	try:
		db = lite.connect(db_filename)
		c = db.cursor()
		c2 = db.cursor()
		c3 = db.cursor()
		c.execute('SELECT DISTINCT fk_article_id FROM sentence')
		for row in c:
			generate_file(c2, c3, row[0])

	except lite.Error as e:
		print("Error %s:" % e.args[0])
		sys.exit(1)
	finally:
		if c:
			c.close()
		if c2:
			c2.close()
		if c3:
			c3.close()
		if db:
			db.close()