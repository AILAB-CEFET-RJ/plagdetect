import tensorflow as tf
from input_helpers import InputHelper
import sqlite3 as lite
import time

tf.flags.DEFINE_integer("batch_size", 1024, "Batch Size (default: 1024)")
tf.flags.DEFINE_integer("percent_dev", 10, "Percentage Dev (default: 10)")
tf.flags.DEFINE_integer("percent_test", 10, "Percentage Dev (default: 10)")
tf.flags.DEFINE_integer("num_docs", 0, "Number of documents to be included on dataset. If 0, then all documents are added (default: 0)")
tf.flags.DEFINE_string("database", '../plag.db', "Database path (default: ../plag.db)")
tf.flags.DEFINE_boolean("auto_chunk", True, "Automatically set chunk_size (default: True")
tf.flags.DEFINE_string("folder", "ds", "Folder in which datasets will be created. (default: ds")
tf.flags.DEFINE_boolean("intra_only", True, "If true, combine sentences of same document only. If false, combines sentences between all documents of a same author. (default: True")

FLAGS = tf.flags.FLAGS

batch_size = FLAGS.batch_size
percent_dev = FLAGS.percent_dev
percent_test = FLAGS.percent_test
database = FLAGS.database
num_docs = FLAGS.num_docs

print("\nParameters:")
for attr, value in sorted(FLAGS.flag_values_dict().iteritems()):
    print("{}={}".format(attr.upper(), value))
print("")

inpH = InputHelper()

start_time = time.time()

db = lite.connect(database)
cursor = db.cursor()
total_count = inpH.my_get_counts(cursor, FLAGS.intra_only, num_docs)

train_count, dev_count, test_count = inpH.build_datasets(cursor, total_count, batch_size, percent_dev, percent_test, FLAGS.auto_chunk, FLAGS.folder, FLAGS.intra_only, num_docs)

end_time = time.time()
print('Time elapsed on dataset creation for {} documents: {} seconds.'.format('all' if num_docs < 0 else num_docs, round(end_time - start_time, 2)))
