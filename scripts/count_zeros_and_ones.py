import os, sqlite3 as lite, time


def build_toy_dataset(c):
	sql = '''select count(*) from dataset_id group by same_style'''
	start_time = time.time()
	c.execute(sql)
	data = c.fetchall()
	print('Query execution time: %s' % (time.time() - start_time))
	with open('same_style.txt', 'w') as f:
		for t in data:
			f.write('\t'.join(str(s) for s in t) + '\n')
	print('Zeros and ones proportion: ', data)


if __name__ == '__main__':
	db_filename = os.join('..', 'plag.db')
	db = lite.connect(db_filename)
	c = db.cursor()
	build_toy_dataset(c)
