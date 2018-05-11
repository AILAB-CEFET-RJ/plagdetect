import sqlite3 as lite
import os, sys, errno, bz2file

def create_folder(directory):
	try:
		os.makedirs(directory)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise

def generate_file(f, doc_id):
	c.execute('SELECT id, fragment, isplag FROM sentence where fk_article_id = ? ORDER BY offset', (doc_id,))
	lines = c.fetchall()
	print('Generating for doc_id ', doc_id)
	for line in lines:
		for line_ahead in lines[lines.index(line)+1:]:
			if line[2] == line_ahead[2] and line[2] == True:
				continue
			same_style = '1' if line[2] == line_ahead[2] else '0'
			f.write(('\t'.join([line[1], line_ahead[1], same_style]) + '\n').encode('utf-8'))
	f.flush()


if __name__ == '__main__':
	global directory
	directory = 'dataset-generated'
	create_folder(directory)

	db_filename = 'plag.db'
	db = None
	try:
		db = lite.connect(db_filename)
		c = db.cursor()
		sql = 'SELECT DISTINCT fk_article_id FROM sentence'
		if len(sys.argv) >= 2:
			sql += ' WHERE fk_article_id >= ' + sys.argv[1]
		if len(sys.argv) >= 3:
			sql += ' and fk_article_id <= ' + sys.argv[2]
		c.execute(sql)
		article_ids = c.fetchall()
		with bz2file.open(os.path.join(directory, 'generated-dataset.bz2'), 'w', 9) as f:
			for article_id in article_ids:
				generate_file(f, article_id[0])

	except lite.Error as e:
		print("Error %s:" % e.args[0])
		sys.exit(1)
	finally:
		if c:
			c.close()
		if db:
			db.close()
