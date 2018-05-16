import sqlite3 as lite
import numpy as np
import os, sys, errno, h5py, pickle, shutil


def create_folder():
	try:
		if os.path.exists(directory):
			print 'Deleting existing directory'
			shutil.rmtree(directory)
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
	sql = 'SELECT DISTINCT author from article'
	c.execute(sql)
	authors = c.fetchall()
	sql = 'SELECT s.id, s.isplag FROM sentence as s INNER JOIN article a ON s.fk_article_id = a.id WHERE author = ?'

	with h5py.File(filename, 'a') as f:
		print 'Generating tuples'
		for author in authors:
			tuples = []
			print 'Progress: ' + str(authors.index(author) + 1) + '/' + str(len(authors))
			c.execute(sql, author)
			sentences = c.fetchall()
			for sentence in sentences:
				for sentence_ahead in sentences[sentences.index(sentence)+1:]:
					if sentence[1] == sentence_ahead[1] and sentence[1] is True:
						continue
					same_style = True if sentence[1] == sentence_ahead[1] else False
					tuples.append((sentence[0], sentence_ahead[0], same_style))
			if 'tuples' in f:
				ds.resize(ds.shape[0]+len(tuples), axis=0)
				ds[-len(tuples):] = np.array(tuples)
				print 'New dataset size: ' + str(ds.shape)
			else:
				ds = f.create_dataset('tuples', maxshape=(None, 3), dtype=int, data=np.array(tuples))
	print 'Tuples generated successfully'


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
