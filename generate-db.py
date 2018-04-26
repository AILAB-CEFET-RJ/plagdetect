import sqlite3 as lite
import xml.etree.ElementTree as et
import sys, os, errno, codecs
import nltk.data


def remove(filename):
	try:
		os.remove(filename)
		print("Old database at file %s removed. Generating new database.", filename)
	except OSError as e: # this would be "except OSError, e:" before Python 2.6
		if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
		 raise # re-raise exception if a different error occurred
		else:
			print("Database file does not exist. Proceeding with database creationg.")

def create_tables(c):
	if c:
		# Create tables
		sql = '''CREATE TABLE IF NOT EXISTS article (
							id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, 
							filename TEXT NOT NULL);'''
		c.execute(sql)

		sql = '''CREATE TABLE IF NOT EXISTS plag (
							id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
							fk_article_id INT,
							fragment TEXT NOT NULL,
							offset INT NOT NULL,
							length INT NOT NULL,
							foreign key (fk_article_id) references article(id));'''
		c.execute(sql)

		sql = '''CREATE TABLE IF NOT EXISTS sentence (
					id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
					fk_article_id INT NOT NULL,
					fragment TEXT NOT NULL,
					offset INT NOT NULL,
					length INT NOT NULL,
					isplag BOOL NOT NULL,
					foreign key (fk_article_id) references article(id));'''
		c.execute(sql)

def insert_into_article_table(c, f):
	sql = '''insert into article(filename) values(?)'''
	c.execute(sql, (f.name,))
	return c.lastrowid

def insert_into_plag_table(c, values):
	sql = '''insert into plag(fk_article_id, fragment, offset, length) values(?,?,?,?)'''
	c.execute(sql, values)

def insert_into_sentence_table(c, article_id, data, sentence, plags):
	sql = '''insert into sentence (fk_article_id, fragment, offset, length, isplag) values (?,?,?,?,?)'''
	values = [article_id, sentence, data.index(sentence), len(sentence)]
	isplag = 0
	for plag_section in plags:
		plag_interval = range(plag_section[0], plag_section[0] + plag_section[1])
		if values[2] in plag_interval and values[2]+values[3] in plag_interval:
			isplag = 1
			break
	values.append(isplag)
	c.execute(sql, values)



def populate_tables(c, tokenizer):
	for file in os.listdir('dataset'):
		txtfile = os.fsdecode(file)
		txtfile = os.path.join('dataset', txtfile) # Get path that works for Windows and Linux
		if txtfile.endswith('.txt'):
			xmlfile = txtfile.replace('.txt', '.xml')
			with open(txtfile, encoding='utf-8-sig', errors='ignore') as f:
				print(f.name)
				article_id = insert_into_article_table(c, f)
				tree = et.parse(xmlfile)
				root = tree.getroot()
				plags = []
				for feature in root:
					if 'name' in feature.attrib and feature.attrib['name'] == 'plagiarism':
						offset = int(feature.attrib['this_offset']) #+ 3
						length = int(feature.attrib['this_length'])
						plags.append((offset, length))
						f.seek(offset)
						data = f.read(length)
						insert_into_plag_table(c, (article_id, data, offset, length))
				f.seek(0)
				data = f.read()
				sentences = tokenizer.tokenize(data)
				for sentence in sentences:
					insert_into_sentence_table(c, article_id, data, sentence, plags)




if __name__ == '__main__':
	db_filename = 'plag.db'
	remove(db_filename)
	db = None
	try:
		db = lite.connect(db_filename)
		c = db.cursor()
		tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

		create_tables(c)
		populate_tables(c, tokenizer)
		db.commit()
	except lite.Error as e:
		print("Error %s:" % e.args[0])
		sys.exit(1)
	finally:
		if db:
			db.close()
