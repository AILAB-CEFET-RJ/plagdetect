import sqlite3 as lite
import xml.etree.ElementTree as et
import sys, os, errno, codecs
import nltk.data
from datagen import dev_train_sep


def remove(filename):
	try:
		os.remove(filename)
		print("Old database at file %s removed. Generating new database.", filename)
	except OSError as e: # this would be "except OSError, e:" before Python 2.6
		if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
		 raise # re-raise exception if a different error occurred
		else:
			print("Database file does not exist. Proceeding with database creation.")

def create_tables(c):
	if c:
		# Create tables
		sql = '''CREATE TABLE IF NOT EXISTS article (
							id INTEGER NOT NULL PRIMARY KEY, 
							filename TEXT NOT NULL,
							fk_author_id INTEGER NOT NULL,
							foreign key (fk_author_id) references author(id));'''
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
					fk_author_id INT NOT NULL,
					fragment TEXT NOT NULL,
					offset INT NOT NULL,
					length INT NOT NULL,
					isplag BOOL NOT NULL,
					foreign key (fk_article_id) references article(id),
					foreign key (fk_author_id) references author(id));'''
		c.execute(sql)

		sql = '''CREATE TABLE IF NOT EXISTS author (
							id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
              name TEXT NOT NULL )'''
		c.execute(sql)

def insert_into_author_table_if_not_exists(c, author):
	sql = '''select a.id from author as a where a.name = ?'''
	c.execute(sql, (author,))
	a = c.fetchall()
	if not len(a) > 0:
		sql = '''insert into author(name) values (?)'''
		c.execute(sql, (author,))
		return c.lastrowid
	return a[0][0]

def insert_into_article_table(c, f, author_id):
	sql = '''insert into article(id, filename, fk_author_id) values(?,?,?)'''
	c.execute(sql, (int(f.name[-9:-4]), f.name, author_id))
	return c.lastrowid

def insert_into_plag_table(c, values):
	sql = '''insert into plag(fk_article_id, fragment, offset, length) values(?,?,?,?)'''
	c.execute(sql, values)

def insert_into_sentence_table(c, article_id, author_id, data, sentence, plags):
	sql = '''insert into sentence (fk_article_id, fk_author_id, fragment, offset, length, isplag) values (?,?,?,?,?,?)'''
	values = [article_id, author_id, sentence.replace('\n', ' ').replace('\r', ' '), data.index(sentence), len(sentence)]
	isplag = 0
	for plag_section in plags:
		plag_interval = range(plag_section[0], plag_section[0] + plag_section[1])
		if values[3] in plag_interval:
			isplag = 1
			break
	values.append(isplag)
	c.execute(sql, values)



def populate_tables(c, tokenizer, datafolder):
	filelist = os.listdir(datafolder)
	filelist.sort()
	for file in filelist:
		file = os.path.join(datafolder, file) # Get path that works for Windows and Linux
		if file.endswith('.txt'):
			xmlfile = file.replace('.txt', '.xml')
			with codecs.open(file, encoding='utf-8-sig') as f:
				data = f.read()
				print(f.name)
				tree = et.parse(xmlfile)
				root = tree.getroot()
				plags = []
				author_is_ignored = False;
				for feature in root:
					if 'authors' in feature.attrib:
						if feature.attrib['authors'] in ignore_list:
							author_is_ignored = True
							break
						author = feature.attrib['authors']
						author_id = insert_into_author_table_if_not_exists(c, author)
						article_id = insert_into_article_table(c, f, author_id)
					elif 'name' in feature.attrib and feature.attrib['name'] == 'plagiarism':
						offset = int(feature.attrib['this_offset']) #+ 3
						length = int(feature.attrib['this_length'])
						plags.append((offset, length))
						insert_into_plag_table(c, (article_id, data[offset:offset+length], offset, length))
				if not author_is_ignored:
					sentences = tokenizer.tokenize(data)
					for sentence in sentences:
						insert_into_sentence_table(c, article_id, author_id, data, sentence, plags)

def create_views(c):
	# Create view showing all authors and number of articles by each one of them.
	sql = '''create view articles_per_author as select author, count(author) as number_of_articles from article group by author;'''
	c.execute(sql)

	sql = '''create view if not exists dataset_id as select s1.id as id1, s2.id as id2, 
			NOT ((s1.isplag AND NOT s2.isplag) OR (NOT s1.isplag AND s2.isplag))
			as same_style FROM sentence as s1, sentence as s2 WHERE 
			(s1.fk_author_id = s2.fk_author_id) AND (s1.id < s2.id) AND NOT (s1.isplag = 1 AND s1.isplag = s2.isplag)'''
	c.execute(sql)

	sql = '''create view if not exists dataset_sentence as select s1.fragment as s1, s2.fragment as s2,
			NOT ((s1.isplag AND NOT s2.isplag) OR (NOT s1.isplag AND s2.isplag)) 
			as same_style FROM sentence as s1, sentence as s2 WHERE
	 		(s1.fk_author_id = s2.fk_author_id) AND (s1.id < s2.id) AND NOT (s1.isplag = 1 AND s1.isplag = s2.isplag)'''
	c.execute(sql)

def create_indexes(c):
	sql = '''CREATE INDEX IF NOT EXISTS article_fk_author_id_index ON article (fk_author_id)'''
	c.execute(sql)
	sql = '''CREATE INDEX IF NOT EXISTS sentence_fk_author_id_index ON sentence (fk_author_id)'''
	c.execute(sql)

def get_ignore_list():
	return [
		'American Tract Society',
		'Consumers\' League of New York, The',
		'Guaranty Trust Company of New York',
		'Teachers of the School Street Universalist Sunday School, Boston',
		'Three Initiates',
		'United States Patent Office',
		'United States.Army.Corps of Engineers.Manhattan District',
		'United States.Congress.House.Committee on Science and Astronautics.',
		'United States.Dept.of Defense',
		'United States.Executive Office of the President',
		'United States.Presidents.',
		'Work Projects Administration',

	]

def help():
	print('''
------------------------------------------------------------------------------------------------------------------------
Usage: \tgenerate-db.py [database file] [dataset folder] [create dataset]

database file:\tpath where database should be created.\t(default = plag.db)
dataset folder:\tpath to folder containing dataset.\t(default = dataset)
create dataset:\tenable table creations for train and dev datasets.\t(default = create)

Note: all paths must be relative to the parent folder of this script.
------------------------------------------------------------------------------------------------------------------------
	''')

if __name__ == '__main__':
	os.chdir('../')

	if len(sys.argv) > 4:
		help()
		raise ValueError('Invalid number of arguments. Received: ' + str(len(sys.argv)-1) + ' Expected: 2')

	if len(sys.argv) > 1:
		db_filename = sys.argv[1]
	else:
		db_filename = 'plag.db'

	if len(sys.argv) > 2:
		datafolder = sys.argv[2]
	else:
		datafolder = 'dataset'

	if len(sys.argv) > 3:
		split_dataset = sys.argv[3]
	else:
		split_dataset = 'create'

	print('Data base will be generated at ')
	remove(db_filename)
	db = None
	global ignore_list
	ignore_list = get_ignore_list()
	try:
		db = lite.connect(db_filename)
		c = db.cursor()
		tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

		create_tables(c)
		populate_tables(c, tokenizer, datafolder)
		create_views(c)
		create_indexes(c)

		if split_dataset == 'create':
			c2 = db.cursor()
			dev_train_sep.separate(c, c2, 100000)

		db.commit()
	except lite.Error as e:
		print("Error %s:" % e.args[0])
		sys.exit(1)
	finally:
		if db:
			db.close()
