import sqlite3 as lite
import os, sys, errno, bz2file

def create_folder(directory):
	try:
		os.makedirs(directory)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise

def generate_file(f, author):
	c.execute('''SELECT s.id, s.fragment, s.isplag FROM sentence as s INNER JOIN article as a 
			ON a.id = s.fk_article_id WHERE a.author = ? ORDER BY a.id, s.offset''', (author,))
	lines = c.fetchall()
	for line in lines:
		for line_ahead in lines[lines.index(line)+1:]:
			if line[2] == line_ahead[2] and line[2] is True:
				continue
			same_style = '1' if line[2] == line_ahead[2] else '0'
			f.write(('\t'.join([line[1], line_ahead[1], same_style]) + '\n').encode('utf-8'))
	f.flush()

def get_sentences_hashmap(c, num_docs):
	sql = 'SELECT id, fragment FROM sentence'
	if(num_docs):
		sql += ' WHERE sentence.fk_article_id <= {}'.format(num_docs)
	c.execute(sql)
	return dict(c.fetchall())

def get_sentences_list(c, num_docs):
	sql = 'SELECT id, fragment FROM sentence'
	if(num_docs):
		sql += ' WHERE sentence.fk_article_id <= {}'.format(num_docs)
	c.execute(sql)
	return c.fetchall()

def get_id_dataset_size():
	sql = '''select count(*) from (select s1.id, s2.id, (s1.isplag = s2.isplag) as same_style FROM
	  					sentence as s1, sentence as s2 WHERE
	  					(s1.fk_author_id = s2.fk_author_id) AND (s1.id < s2.id) AND NOT (s1.isplag = 1 AND s1.isplag = s2.isplag))'''
	c.execute(sql)
	return c.fetchall()

def get_author_ids(c):
	sql = 'SELECT DISTINCT a.id FROM author as a'
	c.execute(sql)

def get_sentence_dataset_by_author_id(c, author_id):
	sql = '''select s1.fragment, s2.fragment, (s1.isplag = s2.isplag) as same_style FROM
  					sentence as s1,
    				sentence as s2 WHERE
  					(s1.fk_author_id = s2.fk_author_id) AND (s1.id < s2.id) AND NOT (s1.isplag = 1 AND s1.isplag = s2.isplag)
  					AND s1.fk_author_id = ?'''
	c.execute(sql, author_id)

def get_id_dataset_by_author_id(c, author_id):
	sql = '''select s1.id, s2.id, (s1.isplag = s2.isplag) as same_style FROM
  					sentence as s1,
    				sentence as s2 WHERE
  					(s1.fk_author_id = s2.fk_author_id) AND (s1.id < s2.id) AND NOT (s1.isplag = 1 AND s1.isplag = s2.isplag)
  					AND s1.fk_author_id = ?'''
	c.execute(sql, author_id)

def get_sentence_dataset(c):
	sql = '''select s1.fragment, s2.fragment, (s1.isplag = s2.isplag) as same_style FROM
  					sentence as s1, sentence as s2 WHERE
  					(s1.fk_author_id = s2.fk_author_id) AND (s1.id < s2.id) AND NOT (s1.isplag = 1 AND s1.isplag = s2.isplag)'''
	c.execute(sql)

def get_id_dataset(c):
	sql = '''select s1.id, s2.id, (s1.isplag = s2.isplag) as same_style FROM
  					sentence as s1, sentence as s2 WHERE
  					(s1.fk_author_id = s2.fk_author_id) AND (s1.id < s2.id) AND NOT (s1.isplag = 1 AND s1.isplag = s2.isplag)'''
	c.execute(sql)

if __name__ == '__main__':
	os.chdir('../')
	global directory
	directory = 'dataset-generated'
	create_folder(directory)

	db_filename = 'plag.db'
	db = None
	try:
		db = lite.connect(db_filename)
		c = db.cursor()
		sql = 'SELECT DISTINCT a.author FROM article as a INNER JOIN sentence as s ON a.id = s.fk_article_id'
		if len(sys.argv) >= 2:
			sql += ' LIMIT ' + sys.argv[1]
		if len(sys.argv) >= 3:
			sql += ' OFFSET ' + sys.argv[2]
		c.execute(sql)
		authors = c.fetchall()
		with bz2file.open(os.path.join(directory, 'generated-dataset.bz2'), 'w', 9) as f:
			i = 1
			for author in authors:
				print 'Progress: ' + str(i) + '/' + str(len(authors)) + ' authors'
				generate_file(f, author[0])
				i += 1

	except lite.Error as e:
		print("Error %s:" % e.args[0])
		sys.exit(1)
	finally:
		if c:
			c.close()
		if db:
			db.close()
