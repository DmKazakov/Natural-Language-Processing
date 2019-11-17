import os
import glob
from pymystem3 import Mystem
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor



mystem = Mystem()
stop_words = set([word.strip() for word in open(os.path.join("resources/russian"), "r")])
class Doc:
    def __init__(self, text):
        words = [word.strip() for words in mystem.lemmatize(text) for word in words.split()
                 if word.strip() and word not in stop_words]
        self.exclamations = sum([word.count("!") for word in words])
        self.questions = sum([word.count("?") for word in words])
        self.open_brackets = sum([word.count("(") for word in words])
        self.close_brackets = sum([word.count(")") for word in words])
        self.words = ' '.join([word for word in words if word.isalnum()])


vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)
def to_vectors(docs):
    tfidf = vectorizer.transform([doc.words for doc in docs])
    remained = []
    for i in range(len(docs)):
        remained.append([docs[i].exclamations, docs[i].questions, docs[i].close_brackets, docs[i].open_brackets])
    remained = csr_matrix(remained)
    print(tfidf.shape)
    return hstack([tfidf, remained]).tocsr()


docs = [Doc(text) for text in open(os.path.join("resources/texts_train.txt"), "r")]
vectorizer.fit([doc.words for doc in docs])
vectors = to_vectors(docs)
labels = [label.strip() for label in open(os.path.join("resources/scores_train.txt"), "r")]
clf = Ridge()
clf.fit(vectors, labels)

test_docs = [Doc(text) for file in glob.glob("resources/dataset.txt") for text in open(os.path.join(file), "r")]
test_vecs = to_vectors(test_docs)
with open("result.txt", "w") as result:
    for label in clf.predict(test_vecs):
        label = max(1, int(round(label)))
        label = min(10, label)
        result.write("%s\n" % label)