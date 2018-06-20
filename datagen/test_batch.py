from generate_dataset import get_dataset_by_author_id
import sqlite3 as lite
import numpy as np
import logging
from keras.utils import np_utils
import pickle

def gerar_triplas_em_lote(image_triples, batch_size, shuffle=False):
    logging.info("Gerando triplas")
    while True:

        # loop once per epoch
        if shuffle:
            indices = np.random.permutation(np.arange(len(image_triples)))
        else:
            indices = np.arange(len(image_triples))
        shuffled_triples = [image_triples[ix] for ix in indices]
        num_batches = len(shuffled_triples) // batch_size

        logging.info("%s batches of %s generated" % (num_batches, batch_size))

        for bid in range(num_batches):
            # loop once per batch
            frase_left, frase_right, labels = [], [], []
            batch = shuffled_triples[bid * batch_size : (bid + 1) * batch_size]
            for i in range(batch_size):
                lhs, rhs, label = batch[i]
                frase_left.append(frases_cache[lhs])
                frase_right.append(frases_cache[rhs])
                labels.append(label)
            Xlhs = frase_left
            Xrhs = frase_right
            Y = np_utils.to_categorical(np.array(labels), num_classes=2)
            yield ([Xlhs, Xrhs], Y)

def generate_dictionary(c):
	sql = 'SELECT id, fragment FROM sentence'
	c.execute(sql)
	return dict(c.fetchall())


################## MAIN ##################
db = lite.connect('plag.db')
c = db.cursor()
frases_cache = generate_dictionary(c)
get_dataset_by_author_id(c, (1,))
triplas = c.fetchmany(1000)

TAMANHO_LOTE = 2

divisor = int(len(triplas) * 0.7)
dados_treino, dados_teste = triplas[0:divisor], triplas[divisor:]

lote_de_treinamento = gerar_triplas_em_lote(dados_treino, TAMANHO_LOTE, shuffle=True)
lote_de_validacao = gerar_triplas_em_lote(dados_teste, TAMANHO_LOTE, shuffle=False)

for value in lote_de_validacao:
    print(str(value[0]) + ' ' + str(value[1]))

print('Fim!')
