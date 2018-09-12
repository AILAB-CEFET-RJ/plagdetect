import os, sqlite3 as lite, time

def build_toy_dataset(c):
	sql = 'select * from dataset_id where same_style = 1 limit 512'
	start_time = time.time()
	c.execute(sql)
	data = c.fetchall()
	print('Query execution time: %s'%(time.time() - start_time))
	with open('same_style.txt', 'w') as f:
		for t in data:
			f.write('\t'.join(str(s) for s in t) + '\n')
	print('Same style tuples retrieved: ', len(data))
	

	sql = 'select * from dataset_id where same_style = 0 limit 512'
	c.execute(sql)
	data = c.fetchall()
	with open('not_same_style.txt', 'w') as f:
		for t in data:
			f.write('\t'.join(str(s) for s in t) + '\n')
	print('Not same style tuples retrieved: ', len(data))


if __name__ == '__main__':
	db_filename = os.join('..', 'plag.db')
	db = lite.connect(db_filename)
	c = db.cursor()
	build_toy_dataset(c)
