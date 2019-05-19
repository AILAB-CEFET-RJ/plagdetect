import sqlite3 as lite
from tensorflow import flags
import matplotlib.pyplot as plt


flags.DEFINE_string('d', '../plag.db', 'Path to database')



FLAGS = flags.FLAGS

print("Connecting to database...")
db = lite.connect(FLAGS.d)
c = db.cursor()
sql = 'select id, fragment from sentence'
tuples = c.execute(sql).fetchall()

print("Counting words...")
with open('words_per_sentence.csv', 'w') as f:
	f.write('\t'.join(('id', 'sentence')) + '\n')
	wcounts = []
	for t in tuples:
		id, sentence = t
		wcount = len(sentence.split())
		wcounts.append(wcount)
		f.write('\t'.join((str(id), str(wcount))) + '\n')
	f.write('max: {}, avg: {}'.format(max(wcounts), float(sum(wcounts)/len(wcounts))))
	
	n, bins, patches = plt.hist(wcounts, 100)
	plt.show()
	#plt.savefig('hist.png')
	