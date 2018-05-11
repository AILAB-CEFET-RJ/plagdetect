import xml.etree.ElementTree as et
import nltk.data

textfile = './dataset/suspicious-document00206.txt'
xmlfile = './dataset/suspicious-document00206.xml'

offsets = dict()

tree = et.parse(xmlfile)
root = tree.getroot()
for feature in root:
	if 'name' in feature.attrib:
		if feature.attrib['name'] == 'plagiarism':
			offset = int(feature.attrib['this_offset'])
			length = int(feature.attrib['this_length'])
			offsets[offset] = length

fp = open(textfile, 'r')
data = fp.read().decode("utf-8-sig").encode("utf-8")

print('\n---file as str:-----------\n\n')
print(data)

print('\n---offset+length-----------\n\n')
print(offsets)

print('\n---plagiarized passages:-----------\n\n')
for offset in offsets:
	start = offset 
	end = offset+offsets[offset]
	passage = data[start:end]
	passage = passage.replace('\n', ' ')
	print(offset, offsets[offset], 'corresponds to:', passage)
	print('\n*****************\n')

print('\n---plagiarized sentences:-----------\n\n')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
for offset in offsets:
	start = offset 
	end = offset+offsets[offset]
	passage = data[start:end]
	passage = passage.replace('\n', ' ')
	print '\n-----\n'.join(tokenizer.tokenize(passage))
	print('\n*****************\n')
