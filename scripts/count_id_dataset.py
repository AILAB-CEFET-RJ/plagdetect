import os, sqlite3 as lite, time


def count_dataset_id(c):
	sql = 'select count(*) from dataset_id'
	start_time = time.time()
	c.execute(sql)
	data = c.fetchall()
	print('Query execution time: %s'%(time.time() - start_time))
	with open('count.txt', 'w') as f:
		f.write(str(data))
	print('Returned rows: ', data)
	

if __name__ == '__main__':
	db_filename = os.join('..', 'plag.db')
	db = lite.connect(db_filename)
	c = db.cursor()
	count_dataset_id(c)
