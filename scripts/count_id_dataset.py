import os, sqlite3 as lite, pickle


def count_dataset_id(c):
	sql = 'select count(*) from dataset_id'
	c.execute(sql)
	data = c.fetchall()
	pickle.dump(data, 'count')
	print('Returned rows: ', data)
	

if __name__ == '__main__':
	os.chdir('../')
	db_filename = 'plag.db'
	db = lite.connect(db_filename)
	c = db.cursor()
	count_dataset_id(c)
