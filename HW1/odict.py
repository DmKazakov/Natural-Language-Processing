import csv
import string
import pandas as pd


class Lemma:
    def __init__(self, value, tag):
        self.value = value
        self.tag = tag

    def __eq__(self, another):
        return hasattr(another, 'value') and hasattr(another, 'tag') and self.value == another.value and self.tag == another.tag

    def __hash__(self):
        return hash(self.value) + hash(self.tag)

    def __repr__(self):
        return f'{self.value}={self.tag}'

    def __str__(self):
        return f'{self.value}={self.tag}'


odcit_tag_map = {
    "межд.": "ADV",
    "част.": "ADV",
    "вводн.": "ADV",
    "предик.": "ADV",
    "н": "ADV",

    "с": "S",
    "со": "S",
    "ж": "S",
    "м": "S",
    "мн.": "S",
    "жо": "S",
    "мо": "S",
    "мо-жо": "S",
    "числ.-п": "S",
    "числ.": "S",

    "св-нсв": "V",
    "нсв": "V",
    "св": "V",

    "п": "A",
    "сравн.": "A",
    "мс-п": "A",

    "предл.": "PR",
    "союз": "CONJ",
}
freq_tag_map = {
    "s": "S",
    "s.PROP": "S",
    "v": "V",
    "a": "A",
    "pr": "PR",
    "conj": "CONJ",
    "spro": "S",
    "adv": "ADV",
    "praedic": "ADV",
    "parenth": "ADV",
    "apro": "A",
    "part": "ADV",
    "advpro": "ADV",
    "praedicpro": "ADV",
    "num": "S",
    "anum": "A",
    "init": "S",
    "intj": "ADV",
    "nonlex": "ADV",
    "com": "ADV"
}
pos_freq = {
    "S":    1431979,
    "V":    826995,
    "PR":   515112,
    "A":    626549,
    "CONJ": 383623,
    "ADV":  253234
}


word_forms = {}
with open("resources/odict.csv", "r", encoding="windows-1251") as csvFile:
    reader = csv.reader(csvFile)
    for forms in reader:
        for form in (forms[:1] + forms[2:]):
            if form:
                form.replace("ё", "е")
                form = form.lower()
                lemmas = word_forms.get(form, set())
                lemmas.add(Lemma(forms[0], odcit_tag_map[forms[1]]))
                word_forms[form] = lemmas
with open("resources/part.txt", "r") as parts:
    for part in parts:
        lemmas = word_forms.get(part, set())
        lemmas.add(Lemma(part, "ADV"))
        word_forms[part] = lemmas
with open("resources/pr.txt", "r") as parts:
    for part in parts:
        lemmas = word_forms.get(part, set())
        lemmas.add(Lemma(part, "PR"))
        word_forms[part] = lemmas
with open("resources/adv.txt", "r") as parts:
    for part in parts:
        lemmas = word_forms.get(part, set())
        lemmas.add(Lemma(part, "ADV"))
        word_forms[part] = lemmas

freq = {}
freq_cvs = pd.read_csv("resources/freq.csv", delimiter='\t')
for freq_stat in freq_cvs.values:
    freq[Lemma(freq_stat[0], freq_tag_map[freq_stat[1]])] = freq_stat[2]


def choose_form(forms):
    lemma = max(forms, key=lambda l: freq.get(l, 0))
    if lemma not in freq:
        return max(forms, key=lambda l: pos_freq[l.tag])
    return lemma


def predict(word):
    if word[:2] == "не" and word[2:] in word_forms:
        lemma = choose_form(word_forms[word[2:]])
        return Lemma("не" + lemma.value, lemma.tag)
    if len(word) <= 2:
        return Lemma(word, "PR")
    if word[-3:] == "ешь" and word[-3:] == "ете" and word[-3:] == "ишь" and word[-3:] == "ите"  and word[-2:] == "ем" \
            and word[-2:] == "им" and word[-2:] == "ет" and word[-2:] == "ут" and word[-2:] == "ют" and word[-2:] == "ит" \
            and word[-2:] == "ат" and word[-2:] == "ят" and word[-2:] == "ся":
        return Lemma(word, "V")
    if word[-2:] == "ие" and word[-2:] == "ые" and word[-2:] == "их" and word[-2:] == "ых" and word[-2:] == "им" \
            and word[-2:] == "им" and word[-2:] == "ым" and word[-2:] == "ие" and word[-2:] == "ые" and word[-2:] == "ие" \
            and word[-3:] == "ими" and word[-3:] == "ыми" and word[-2:] == "их" and word[-2:] == "ых" and word[-2:] == "ые" \
            and word[-2:] == "ой" and word[-3:] == "ого" and word[-3:] == "ому" and word[-2:] == "ой" and word[-3:] == "ого"\
            and word[-2:] == "ым" and word[-2:] == "ом" and word[-2:] == "ое" and word[-2:] == "ая" and word[-2:] == "ой" \
            and word[-2:] == "ий" and word[-3:] == "его" and word[-3:] == "ему" and word[-2:] == "им" and word[-2:] == "ем"\
            and word[-2:] == "ее" and word[-2:] == "яя" and word[-2:] == "ей" and word[-2:] == "юю":
        return Lemma(word, "A")
    return Lemma(word, "S")


with open("resources/dataset.txt", "r") as dataset_file:
    with open("result.txt", "w") as result:
        for line in dataset_file:
            result_line = ""
            for word in line.split():
                word = word.translate(str.maketrans('', '', string.punctuation))
                lowered = word.lower()
                lowered.replace("ё", "е")
                lemma = Lemma(word, "S")
                if lowered in word_forms:
                    lemma = choose_form(word_forms[lowered])
                else:
                    lemma = predict(lowered)
                result_line += f'{word}{{{lemma.value}={lemma.tag}}} '
            result.write(result_line[:-1] + "\n")
