import tensorflow as tf
from input_helpers import InputHelper
import sqlite3 as lite
import time

tf.flags.DEFINE_integer("batch_size", 1024, "Batch Size (default: 1024)")
tf.flags.DEFINE_integer("percent_dev", 10, "Percentage Dev (default: 10)")
tf.flags.DEFINE_integer("percent_test", 10, "Percentage Dev (default: 10)")
tf.flags.DEFINE_string("database", '../plag1.db', "Database path (default: ../plag.db)")

FLAGS = tf.flags.FLAGS

inpH = InputHelper()
batch_size = FLAGS.batch_size
percent_dev = FLAGS.percent_dev
percent_test = FLAGS.percent_test
database = FLAGS.database

# start_time = time.time()

db = lite.connect(database)
cursor = db.cursor()
total_count = inpH.my_get_counts(cursor)

train_count, dev_count, test_count = inpH.build_datasets(cursor, total_count, batch_size, percent_dev, percent_test)

# end_time = time.time()
# print('Time elapsed on dataset creation: {} seconds.'.format(round(end_time - start_time, 2)))
