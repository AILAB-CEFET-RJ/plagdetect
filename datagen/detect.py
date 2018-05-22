import glob, chardet, os

os.chdir('../')
f = open('./dataset/suspicious-document00206.txt', mode='rb')
chardet.detect(f)
