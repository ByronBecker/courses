from csv import DictReader, DictWriter

import numpy as np
from numpy import array

from sklearn.pipeline import FeatureUnion, Pipeline, make_union
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.stem.porter import PorterStemmer

from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

kTARGET_FIELD = 'spoiler'
kTEXT_FIELD = 'sentence'
kPAGE_FIELD = 'page'
kKEYWORDS_FIELD = 'keywords'
kTROPE_FIELD = 'trope'

stemmer = nltk.PorterStemmer()

keys = ["season", "die", "died", "dies", "death", "suicide", "kills", "killed", "baby", "pregnant", "child", "destiny", "cried", "revealed", "reveals", "will", "believe", "spoiler", "shocked", "shocking", "episode", "finale", "end", "turns"]

death = {'sentence': 'die died dies death suicide kills killed end', 'spoiler': 'True', 'trope': 'TheBest'}
#death1 = {'sentence': 'die died dies death suicide kills killed end', 'spoiler': 'True', 'trope': 'TheBest'}
#death2 = {'sentence': 'die died dies death suicide kills killed end', 'spoiler': 'True', 'trope': 'TheBest'}
surprise = {'sentence': 'revealed reveals spoiler shocked shocking turns', 'spoiler': 'True', 'trope': 'TheBest'}
extras = {'sentence': 'season baby pregnant destiny cried will believe finale', 'spoiler': 'True','trope': 'TheBest'}



#ItemSelector source: http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html
class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
    'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
    The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]
    
FeatureUnion

class Featurizer:
    def __init__(self):
        #estimators = [("bag-of-words", CountVectorizer(analyzer='word', strip_accents='ascii', stop_words='english', ngram_range=(1,3))), ("tfidf", TfidfVectorizer(analyzer = 'word', max_df=0.7, ngram_range=(1,3), smooth_idf=True))]
        #self.vectorizer = FeatureUnion(estimators)
        #self.vectorizer = TfidfVectorizer(strip_accents='ascii', stop_words='english')
        #self.vectorizer = CountVectorizer()
        #self.vectorizer = CountVectorizer(analyzer='word', strip_accents='ascii', stop_words='english', ngram_range=(1,8))
        self.vectorizer = TfidfVectorizer(analyzer='word', strip_accents='ascii', stop_words='english', ngram_range=(1,3), smooth_idf=True ) 
        '''
        self.vectorizer = Pipeline([
            ('union', FeatureUnion(
                transformer_list=[

                    ('sentence', Pipeline([
                        ('selector', ItemSelector(key='sentence')),
                        ('tfidf', TfidfVectorizer())
                    ])),
                
                    ('trope', Pipeline([
                        ('selector', ItemSelector(key='trope')),
                        ('tfidf', TfidfVectorizer()),
                    ])),
                
                ],

            transformer_weights={
                'text': 1.0,
                #'trope':0.4,
            },
            )),

            #('svc', SVC(kernel='linear')),
        ])
        '''
            

        '''
        FeatureUnion

        column, Pipeline
            column, Itemselector(key = column)
            name of vectorizer, vectorize (params)
        '''
        '''
                        FeatureUnion([
            ("b-o-w",CountVectorizer(
                analyzer='word', 
                strip_accents='ascii',  #strip accents w/ascii      **try unicode for final submission
                stop_words='english',   #remove stop words
                ngram_range=(1,3)
                )), 
            ("tfidf", TfidfVectorizer(
                analyzer = 'word',
                max_df=0.7,     #ignore words w/high freq
                ngram_range=(1,3)
                #smooth_idf=True
                ))
            ]) 
        '''
        #self.vectorizer = CountVectorizer(analyzer='word', strip_accents='ascii', stop_words='english', ngram_range=(1,3)) 
        #after changed ngram range, training accuracy went from 0.64 to 0.46

        #self.vectorizer = make_union(CountVectorizer(), TfidfTransformer()) 
        #self.vectorizer = FeatureUnion([('countvec', CountVectorizer()), ('tfidf', TfidfTransformer())])

    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)  #fits weights based on training data

    def test_feature(self, examples):
        return self.vectorizer.transform(examples) #transforms in test data to the specifications (for the test file)

    def show_top10(self, classifier, categories):
        #after fitting, runs and returns the top and bottom 10 for the 0 class (not a spoiler)
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        if len(categories) == 2:
            top10 = np.argsort(classifier.coef_[0])[-10:]   
            bottom10 = np.argsort(classifier.coef_[0])[:10]
            print("Pos: %s" % " ".join(feature_names[top10]))
            print("Neg: %s" % " ".join(feature_names[bottom10]))
        else:
            for i, category in enumerate(categories):
                top10 = np.argsort(classifier.coef_[i])[-10:]
                print("%s: %s" % (category, " ".join(feature_names[top10])))

if __name__ == "__main__":

    # Cast to list to keep it all in memory
    train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
    test = list(DictReader(open("../data/spoilers/test.csv", 'r')))
    '''
    def tokenize(train):
        tokens = nltk.word_tokenize(text)
        stems = stem_tokens(tokens, stemmer)
        return stems
    '''


    feat = Featurizer()


    labels = []
    for line in train:      #puts spoiler in all labels if not there already
        if not line[kTARGET_FIELD] in labels:
            labels.append(line[kTARGET_FIELD])
        #if not line[kTROPE_FIELD] in labels:
        #    labels.append(line[kTROPE_FIELD])


    #train.append(death)
    #train.append(death)
    #train.append(death)
    #train.append(surprise)
    #train.append(extras)
    #train.append({'sentence': 'season died dies death suicide kills killed baby pregnant child destiny tear', 'spoiler': 'True', 'trope': 'TheBest'})
    #train.append({'sentence': 'happens will believe spoiler shocked shocking', 'spoiler': 'True'}) 
    #print train

    #for x in train:
        #print
    #train_trope = [x[kTROPE_FIELD] for x in train]
    #train_text = [x[kTEXT_FIELD] for x in train]
    #test_trope = [x[kTROPE_FIELD] for x in train]
    #test_text = [x[kTEXT_FIELD] for x in train]
    #tropes = []
    #for i in xrange(1, len(l)):
        #print l[i]
    #trope = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False).fit_transform(l) 
    #trope = CountVectorizer()

    #x_trope = trope.fit_transform(l)

    #print x_trope

    #fit_train = train.fit_transform(train)
    #idf_trans = TfidfTransformer(use_idf=True).fit(x_train)
    #x1_train = idf_trans.fit_transform(x_train)
    #x2_train = 
    '''
    allfeat = FeatureUnion([
        ("b-o-w",CountVectorizer(
            analyzer='word', 
            strip_accents='ascii',  #strip accents w/ascii      **try unicode for final submission
            stop_words='english',   #remove stop words
            ngram_range=(1,3))
        ), 
        ("tfidf", TfidfVectorizer(
            analyzer = 'word',
            max_df=0.7,     #ignore words w/high freq
            ngram_range=(1,3),
            smooth_idf=True)
        )
        ])

    all_train = allfeat.fit_transform([x[kTEXT_FIELD] for x in train])
    print all_train
    '''

    print("Label set: %s" % str(labels))
    x_train = feat.train_feature([x[kTEXT_FIELD] for x in train])

                                  #y[kTROPE_FIELD] for y in train])     
    #makes fits features of each line or "text field"
    x_test = feat.test_feature([x[kTEXT_FIELD] for x in train])

    #y[kTROPE_FIELD] for y in test])  #runs the test_features after this of each line
    #idf = TfidfVectorizer()
    #x1_train = idf.fit_transform(x[kTEXT_FIELD] for x in train)
    #for x in x1_train:
    #    print x
    #print x1_train
    #print(x_train)


    #all_feat = FeatureUnion([
        #("bag-of-words", 


    '''
    allfeat = FeatureUnion([
        ("b-o-w",CountVectorizer(
            analyzer='word', 
            strip_accents='ascii',  #strip accents w/ascii      **try unicode for final submission
            stop_words='english',   #remove stop words
            ngram_range=(1,3)), 
        ("tfidf", TfidfVectorizer(
            analyzer = 'word',
            max_df=0.7,     #ignore words w/high freq
            ngram_range=(1,3)
            smooth_idf=True
    '''


    y_train = array(list(labels.index(x[kTARGET_FIELD])     #looks at the spoilers label for each line in train
                         for x in train))
    #for i in x_train:
    #    print(i)
    #print x_train
    #x_train.append("season died dies suicide kills killed, baby")

    #idf_trans = TfidfVectorizer(analyzer='word', strip_accents='ascii',ngram_range=(1,4),stop_words='english')
    


    #x1_train = idf_trans.fit_transform(x_train)


    #print(x1_train)

    print(len(train), len(y_train))
    print(set(y_train))

    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)  #loss - log "logistic reg" penalty - l2 "ridge" regression"
    lr.fit(x_train, y_train)    #fits the linear model with SGD

    #print(lr)

    feat.show_top10(lr, labels)

    predictions = lr.predict(x_test)
    o = DictWriter(open("predictions.csv", 'w'), ["Id", "spoiler"])
    o.writeheader()
    for ii, pp in zip([x['Id'] for x in test], predictions):
        d = {'Id': ii, 'spoiler': labels[pp]}
        o.writerow(d)
    '''
    trainpredictions = lr.predict(feat.test_feature(x[kTEXT_FIELD] for x in train))
    o = DictWriter(open("predictionstrain.csv", 'w'), ["Id", "spoiler"])
    o.writeheader()
    i = 0
    for ii, pp in zip([i for x in train], predictions):
        d = {'Id': ii, 'spoiler': labels[pp]}
        i = i + 1
        o.writerow(d)
   ''' 
    
