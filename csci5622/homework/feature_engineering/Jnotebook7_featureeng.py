from collections import defaultdict
import string
import operator

import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import accuracy_score

import nltk
nltk.download('wordnet')
nltk.download('brown')

from nltk.corpus import wordnet as wn
from nltk.corpus import brown
from nltk.util import ngrams

import seaborn as sn 
import matplotlib.pyplot as plt
%matplotlib inline

mycolors = {"blue": "steelblue", "red": "#a76c6e", "green": "#6a9373"}


def normalize_tags(tag):
    if not tag or not tag[0] in string.uppercase:
        return "PUNC"
    else:
        return tag[:2]


kTAGSET = ["", "JJ", "NN", "PP", "RB", "VB"]

class Analyzer:
    def __init__(self, word, before, after, prev, next, char, dict):
        self.word = word
        self.after = after
        self.before = before
        self.prev = prev
        self.next = next
        self.dict = dict
        self.char = char

    def __call__(self, feature_string):
        feats = feature_string.split()

    if self.word:
        yield feats[0]

    if self.after:
        for ii in [x for x in feats if x.startswith("A:")]:
            yield ii
    if self.before:
        for ii in [x for x in feats if x.startswith("B:")]:
            yield ii
    if self.prev:
        for ii in [x for x in feats if x.startswith("P:")]:
            yield ii
    if self.next:
        for ii in [x for x in feats if x.startswith("N:")]:
            yield ii
    if self.dict:
        for ii in [x for x in feats if x.startswith("D:")]:
            yield ii
    if self.char:
        for ii in [x for x in feats if x.startswith("C:")]:
            yield ii

def example(sentence, position):
    word = sentence[position][0]
    ex = word
    tag = normalize_tags(sentence[position][1])
    if tag in kTAGSET:
        target = kTAGSET.index(tag)
    else:
        target = None

    if position > 0:
        prev = " P:%s" % sentence[position - 1][0]
    else:
        prev = ""

    if position < len(sentence) - 1:
        next = " N:%s" % sentence[position + 1][0]
    else:
        next = ''

    all_before = " " + " ".join(["B:%s" % x[0]
        for x in sentence[:position]])
    all_after = " " + " ".join(["A:%s" % x[0]
        for x in sentence[(position + 1):]])

    dictionary = ["D:ADJ"] * len(wn.synsets(word, wn.ADJ)) + \
    ["D:ADV"] * len(wn.synsets(word, wn.ADV)) + \
    ["D:VERB"] * len(wn.synsets(word, wn.VERB)) + \
    ["D:NOUN"] * len(wn.synsets(word, wn.NOUN))

    dictionary = " " + " ".join(dictionary)

    char = ' '
    padded_word = "~%s^" % sentence[position][0]
    for ngram_length in xrange(2, 5):
        char += ' ' + " ".join("C:%s" % "".join(cc for cc in x)
            for x in ngrams(padded_word, ngram_length))
    ex += char

    ex += prev
    ex += next
    ex += all_after
    ex += all_before
    ex += dictionary

    return ex, target

def all_examples(limit, train=True):
    sent_num = 0
    for ii in brown.tagged_sents():
        sent_num += 1
        if limit > 0 and sent_num > limit:
            break

        for jj in xrange(len(ii)):
            ex, tgt = example(ii, jj)
            if tgt:
                if train and sent_num % 5 != 0:
                    yield ex, tgt
                if not train and sent_num % 5 == 0:
                    yield ex, tgt

def accuracy(classifier, x, y, examples):
    predictions = classifier.predict(x)
    cm = confusion_matrix(y, predictions)

    print("Accuracy: %f" % accuracy_score(y, predictions))

    print("\t".join(kTAGSET[1:]))
    for ii in cm:
        print("\t".join(str(x) for x in ii))

    errors = defaultdict(int)
    for ii, ex_tuple in enumerate(examples):
        ex, tgt = ex_tuple
        if tgt != predictions[ii]:
            errors[(ex.split()[0], kTAGSET[predictions[ii]])] += 1

    for ww, cc in sorted(errors.items(), key=operator.itemgetter(1),
    reverse=True)[:10]:
        print("%s\t%i" % (ww, cc))

def part_of_speech(**kwargs):
    word = kwargs.get("word", False)
    all_before = kwargs.get("all_before", False)
    all_after = kwargs.get("all_after", False)
    one_before = kwargs.get("one_before", False)
    one_after = kwargs.get("one_after", False)
    characters = kwargs.get("characters", False)
    dictionary = kwargs.get("dictionary", False)
    limit= kwargs.get("limit",-1)

    analyzer = Analyzer(word, all_before, all_after,
    one_before, one_after, characters,
    dictionary)

    vectorizer = HashingVectorizer(analyzer=analyzer)

    x_train = vectorizer.fit_transform(ex for ex, tgt in
    all_examples(limit))
    x_valid = vectorizer.fit_transform(ex for ex, tgt in
    all_examples(limit, train=False))

    for ex, tgt in all_examples(1):
        print(" ".join(analyzer(ex)))

    y_train = np.array(list(tgt for ex, tgt in all_examples(limit)))
    y_valid = np.array(list(tgt for ex, tgt in
    all_examples(limit, train=False)))

    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True) 
#doing standard multiclass SGD with L2 regularization
    lr.fit(x_train, y_train)

    print("TRAIN\n-------------------------")
    accuracy(lr, x_train, y_train, all_examples(limit))
    print("Validation\n--------------------")
    accuracy(lr, x_valid, y_valid, all_examples(limit, train=False))

np.random.seed(1234)

def income_data(N=200):
    x = 1.1*np.random.normal(size=N)
    y = np.exp(x)
    return y
'''
from IPython.core.display import HTML
HTML("""
<style>
.MathJax nobr>span.math>span{border-left-width:0 !important};
</style>
""")
'''
