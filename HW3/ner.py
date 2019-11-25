import os
import glob
from enum import Enum

import lxml.etree as et
from nltk.tokenize import RegexpTokenizer
from pymystem3 import Mystem


class Tag(Enum):
    NONE = 0,
    ORG = 1,
    PERSON = 2


tag_map = {
    "Name": Tag.PERSON,
    "Surn": Tag.PERSON,
    "Patr": Tag.PERSON,
    "Orgn": Tag.ORG,
    "Trad": Tag.ORG,
}


class Node:
    def __init__(self, tag=Tag.NONE):
        self.tag = tag
        self.edges = {}

    def get_first_match(self, words):
        if self.tag != Tag.NONE:
            return self.tag, 0
        if len(words) == 0:
            return Tag.NONE, 0
        if words[0] in self.edges:
            tag, size = self.edges[words[0]].get_first_match(words[1:])
            return tag, size + 1 if tag != Tag.NONE else 0
        return Tag.NONE, 0

    def add(self, words, tag):
        if len(words) == 0:
            self.tag = tag
            return
        if words[0] not in self.edges:
            self.edges[words[0]] = Node()
        self.edges[words[0]].add(words[1:], tag)

    def add_all(self, words, tag):
        for word in words:
            if word not in self.edges:
                self.edges[word] = Node()
            self.edges[word].tag = tag


mystem = Mystem()
tokenizer = RegexpTokenizer(r'((?:[.\-"]?\w[.\-"]?)+)')
root = Node()
def lemmatize(word):
    return "".join([s.strip() for s in mystem.lemmatize(word)])


def parse_dict(org_tag, per_tag, files, words):
    for train_file_name in files:
        with open(os.path.join(train_file_name), "r") as train_file:
            for line in train_file:
                tag = line.split()[1]
                word = words(line)
                if tag == org_tag:
                    if len(word) == 1 and (word[0] == "большой"):
                        continue
                    root.add(word, tag)
                elif tag == per_tag:
                    root.add_all(word, tag)


def preprocess(s):
    return s.replace('«', '"').replace('»', '"').replace('ё', 'e')


def tokenize1(s):
    s = preprocess(s)
    return tokenizer.tokenize(s)[4:]


def tokenize2(s):
    s = preprocess(s)
    s = s.split("#")[1]
    return tokenizer.tokenize(s)


def predict(s):
    return 0


parse_dict("ORG", "PER", glob.glob("resources/Collection5/*.ann"), lambda s : tokenize1(s))
parse_dict("Org", "Person", glob.glob("resources/testset/*.objects"), lambda s : tokenize2(s))
context = et.iterparse("resources/dict.opcorpora.xml", tag='lemma')
for (_, element) in context:
    tag = Tag.NONE
    lemma = element[0]
    for g in lemma:
        if g.attrib['v'] in tag_map:
            tag = tag_map[g.attrib['v']]
    if tag != Tag.NONE:
        for form in element[1:]:
            root.add([form.attrib['t']], tag)

with open("result.txt", "w") as result_file:
    with open("resources/dataset_40163_1.txt", "r") as dataset:
        for sentence in dataset:
            sentence = preprocess(sentence)
            tokens = [lemmatize(token) for token in tokenizer.tokenize(sentence)]
            positions = list(tokenizer.span_tokenize(sentence))
            current_index = 0
            while current_index < len(tokens):
                (tag, size) = root.get_first_match(tokens[current_index:])
                #if tag == Tag.NONE:
                #    (tag, size) = predict(tokens[current_index:])
                for index in range(current_index, current_index + size):
                    result_file.write(f"{positions[index][0]} {positions[index][1] - positions[index][0]} {tag.name} ")
                if tag == Tag.NONE:
                    size = 1
                current_index += size
            result_file.write("EOL\n")
