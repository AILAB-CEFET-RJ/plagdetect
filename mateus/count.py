import os


os.chdir('../')
with open('../pan11-intrinsic-plagiarism-detection-test-corpus-2011-05-13/suspicious-documents/suspicious-document03103.txt') as infile:
    lines=0
    words=0
    characters=0
    for line in infile:
        wordslist=line.split()
        lines=lines+1
        words=words+len(wordslist)
        characters += sum(len(word) for word in wordslist)
print(lines)
print(words)
print(characters)
