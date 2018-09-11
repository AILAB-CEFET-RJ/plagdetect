import os, sqlite3 as lite

def build_toy_dataset(c):
	sql = 'select * from dataset_id where same_style = 1 limit 500'
	c.execute(sql)
	data = c.fetchall()
	with open('same_style.txt', 'w') as f:
		for t in data:
			f.write('\t'.join(str(s) for s in t) + '\n')
	print('Same style tuples retrieved: ', len(data))
	

	sql = 'select * from dataset_id where same_style = 0 limit 500'
	c.execute(sql)
	data = c.fetchall()
	with open('not_same_style.txt', 'w') as f:
		for t in data:
			f.write('\t'.join(str(s) for s in t) + '\n')
	print('Not same style tuples retrieved: ', len(data))


if __name__ == '__main__':
	db_filename = '../plag.db'
	db = lite.connect(db_filename)
	c = db.cursor()
	build_toy_dataset(c)
