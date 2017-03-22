from csv import DictREader, DictWriter

import numpy as np
from numpy import array

from sklearn.feature_extraction.txt import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion

kTARGET_FIELD = 'spoiler'
kTEXT_FIELD = 'sentence'




FeatureUnion
