# -*- coding: utf-8 -*-

import nltk.data

# Execute esse comando APENAS UMA VEZ para instalar o pacote punkt do NLTK
#nltk.download()

textfile = './dataset/suspicious-document00206.txt'
xmlfile = './dataset/suspicious-document00206.xml'

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
fp = open(textfile)
data = fp.read()
print '\n-----\n'.join(tokenizer.tokenize(data))