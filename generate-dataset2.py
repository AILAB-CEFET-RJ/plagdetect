import sqlite3 as lite
import os, sys, errno, bz2file, pickle


def create_folder():
	try:
		os.makedirs(directory)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise


def generate_dictionary(c):
	filename = os.path.join(directory, 'hashmap')
	sql = 'SELECT id, fragment FROM sentence'
	c.execute(sql)
	hash_map = dict(c.fetchall())

	print 'Generating dictionary'
	with open(filename, 'w') as f:
		pickle.dump(hash_map, f)
	print 'Dictionary generated successfully.'


def generate_tuples(c):
	filename = os.path.join(directory, 'tuples')
	sql = 'SELECT DISTINCT author from article LIMIT 1'
	c.execute(sql)
	authors = c.fetchall()
	tuples = []
	sql = 'SELECT s.id, s.isplag FROM sentence as s INNER JOIN article a ON s.fk_article_id = a.id WHERE author = ?'

	print 'Generating tuples'
	for author in authors:
		print 'Progress: ' + str(authors.index(author) + 1) + '/' + str(len(authors))
		c.execute(sql, author)
		sentences = c.fetchall()
		for sentence in sentences:
			for sentence_ahead in sentences[sentences.index(sentence)+1:]:
				if sentence[1] == sentence_ahead[1] and sentence[1] is True:
					continue
				same_style = True if sentence[1] == sentence_ahead[1] else False
				tuples.append((sentence[0], sentence_ahead[0], same_style))

	print 'Tuples generated successfully'
	print 'Dumping tuples'
	with open(filename, 'w') as f:
		pickle.dump(tuples, f)
	print 'Tuples dumped successfully'


if __name__ == '__main__':
	global directory
	if len(sys.argv) > 1:
		directory = sys.argv[1]
	else:
		directory = 'dataset-generated2'
	create_folder()

	db_filename = 'plag.db'
	db = None
	try:
		db = lite.connect(db_filename)
		c = db.cursor()
		generate_dictionary(c)
		generate_tuples(c)

	except lite.Error as e:
		print("Error %s:" % e.args[0])
		sys.exit(1)
	finally:
		if c:
			c.close()
		if db:
			db.close()
