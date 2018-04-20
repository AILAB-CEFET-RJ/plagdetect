import xml.etree.ElementTree as et

textfile = 'suspicious-document00001.txt'
xmlfile = 'suspicious-document00001.xml'

with open(textfile) as f:
	tree = et.parse(xmlfile)
	root = tree.getroot()
	for feature in root:
		if 'name' in feature.attrib:
			if feature.attrib['name'] == 'plagiarism':
				start = int(feature.attrib['this_offset'])
				offset = int(feature.attrib['this_length'])
				f.seek(start)
				data = f.read(offset)
				print('Data:\n')
				print(data)
				print('-----------\n\n')
