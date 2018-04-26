# -*- coding: utf-8 -*-

import nltk.data, codecs

# Execute esse comando APENAS UMA VEZ para instalar o pacote punkt do NLTK
#nltk.download()

textfile = './dataset/suspicious-document00206.txt'
xmlfile = './dataset/suspicious-document00206.xml'

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
fp = codecs.open(textfile, encoding='utf-8')
data = fp.read()
a = tokenizer.tokenize(data)
print ('\n-----\n'.join(a))