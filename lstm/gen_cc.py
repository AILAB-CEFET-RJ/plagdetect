import tensorflow as tf
from input_helpers import InputHelper
import sqlite3 as lite
import os, shutil, random

#Fixing random for reproducibility
random.seed(1)

tf.flags.DEFINE_string("database", '../plag.db', "Database path (default: ../plag.db)")
tf.flags.DEFINE_string("output_dir", '../clusters_correlation/data', "Path where files will be generated (default: ../clusters_correlation/data)")

FLAGS = tf.flags.FLAGS

print("\nParameters:")
for attr, value in sorted(FLAGS.flag_values_dict().iteritems()):
    print("{}={}".format(attr.upper(), value))
print("")


# returns a list of tuples, each containing the id for each document found in the database.
def get_document_ids(cursor):
	sql = 'select id from article'
	cursor.execute(sql)
	return cursor.fetchall()



########## main ##########
inpH = InputHelper()
db = lite.connect(FLAGS.database)
cursor = db.cursor()

doc_ids = get_document_ids(cursor)
doc_count = len(doc_ids)

if os.path.exists(FLAGS.output_dir):
	shutil.rmtree(FLAGS.output_dir, ignore_errors=True)
os.mkdir(FLAGS.output_dir)

i = 0
for doc_id in doc_ids:
	sql = 'select count(*) from sentence where fk_article_id = ?'
	cursor.execute(sql, doc_id)
	tuple_count = cursor.fetchall()[0][0]

	sql = '''select s1.id as id1, s2.id as id2 FROM sentence as s1, sentence as s2 WHERE 
			(s1.fk_article_id = s2.fk_article_id) AND (s1.id < s2.id)	and s1.fk_article_id = ?'''
	cursor.execute(sql, doc_id)
	sentence_ids = cursor.fetchall()

	# TODO load embeddings, feed them to trained model and use results for 3rd element on tuple

	with open(FLAGS.output_dir + '/' + str(doc_id[0]) + '.txt', 'w+') as f:
		i = i + 1
		print 'Building {} ({}/{})'.format(f.name, i, doc_count)
		f.writelines(['5\n', '100\n', str(tuple_count) + '\n', '100\n'])
		lines = []
		for id_tuple in sentence_ids:
			# TODO replace random for model prediction
			line = '  '.join((str(id_tuple[0]), str(id_tuple[1]), str(random.random() - 0.5))) + '\n'
			lines.append(line)

		f.writelines(lines)


