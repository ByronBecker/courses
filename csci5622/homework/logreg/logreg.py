import random
import argparse
import numpy as np
from numpy import zeros, sign 
from math import exp, log, floor
from collections import defaultdict


kSEED = 1735
kBIAS = "BIAS_CONSTANT"

random.seed(kSEED)


def sigmoid(score, threshold=20.0):
    """
    Prevent overflow of exp by capping activation at 20.

    :param score: A real valued number to convert into a number between 0 and 1
    """

    if abs(score) > threshold:
        score = threshold * sign(score)

    return 1.0 / (1.0 + exp(-score))


class Example:
    """
    Class to represent a logistic regression example
    """
    def __init__(self, label, words, vocab, df):
        """
        Create a new example

        :param label: The label (0 / 1) of the example (i.e. binary classifier)
        :param words: The words in a list of "word:count" format (words in the doc) 
        :param vocab: The vocabulary to use as features (list) (features)
        """
        self.nonzero = {}
        self.y = label
        self.x = zeros(len(vocab))  #initialize weights
        for word, count in [x.split(":") for x in words]:
            if word in vocab:
                assert word != kBIAS, "Bias can't actually appear in document"
                self.x[vocab.index(word)] += float(count)   #increment the count of the word
                self.nonzero[vocab.index(word)] = word      #put the value of that count into a dictionary (i.e. nonzero terms there)
        self.x[0] = 1   #initial w0 term = 1


class LogReg:
    def __init__(self, num_features, lam, eta=lambda x: 0.1):       
        """
        Create a logistic regression classifier

        :param num_features: The number of features (including bias)
        :param lam: Regularization parameter
        :param eta: A function that takes the iteration as an argument (the default is a constant value)
        """
        
        self.w = zeros(num_features)    #create a list of zeros w/the number of features
        self.lam = lam                  #regularization parameter 
        self.eta = eta                  #set to 0.1
        self.last_update = defaultdict(int)

        assert self.lam>= 0, "Regularization parameter must be non-negative"

    def progress(self, examples):
        """
        Given a set of examples, compute the probability and accuracy

        :param examples: The dataset to score
        :return: A tuple of (log probability, accuracy)
        """

        logprob = 0.0
        num_right = 0
        for ex in examples:
            p = sigmoid(self.w.dot(ex.x))   #calc the sigmoid of the feature vector evaluated at example xs
            if ex.y == 1:
                logprob += log(p)       
            else:
                logprob += log(1.0 - p)

            # Get accuracy
            if abs(ex.y - p) < 0.5:
                num_right += 1      # if within 0.5, model is correct

        return logprob, float(num_right) / float(len(examples))



    def sg_update(self, train_example, iteration, use_tfidf=False):
        """
        Compute a stochastic gradient update to improve the log likelihood.

        :param train_example: The example to take the gradient with respect to
        :param iteration: The current iteration (an integer)
        :param use_tfidf: A boolean to switch between the raw data and the tfidf representation
        :return: Return the new value of the regression coefficients
        """
        
        
        # TODO: Implement updates in this function
        if iteration > 0:
            self.w == self.last_update[iteration-1]
        wtx = np.dot(self.w, np.transpose([train_example.x]))    #matrix multiply xs (1x5) by wT (5x1) to get 1x1
        #put brackets around self.w to make it a 2d array so could transpose it

        delta = np.dot((train_example.y - sigmoid(wtx)), train_example.x)   # (y - wtx) * x
        self.w = self.w + self.eta(iteration)*delta

        #Implement regularization
        shrinkage = np.array(2 * self.eta(iteration) *self.lam * self.w)
        shrinkage[0] = 0  # don't bias w0
        self.w = self.w - shrinkage
        self.last_update[iteration] = self.w

        return self.w
        

def eta_schedule(iteration):
    # TODO (extra credit): Update this function to provide an
    # EFFECTIVE iteration dependent learning rate size.  
    return 1.0 

def read_dataset(positive, negative, vocab, test_proportion=0.1):
    """
    Reads in a text dataset with a given vocabulary

    :param positive: Positive examples
    :param negative: Negative examples
    :param vocab: A list of vocabulary words
    :param test_proprotion: How much of the data should be reserved for test
    """

    df = [float(x.split("\t")[1]) for x in open(vocab, 'r') if '\t' in x]
    vocab = [x.split("\t")[0] for x in open(vocab, 'r') if '\t' in x]
    assert vocab[0] == kBIAS, \
        "First vocab word must be bias term (was %s)" % vocab[0]

    train = []
    test = []
    for label, input in [(1, positive), (0, negative)]:
        for line in open(input):
            ex = Example(label, line.split(), vocab, df)
            if random.random() <= test_proportion:
                test.append(ex)
            else:
                train.append(ex)

    # Shuffle the data 
    random.shuffle(train)
    random.shuffle(test)

    return train, test, vocab



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--lam", help="Weight of L2 regression",
                           type=float, default=0.0, required=False)
    argparser.add_argument("--eta", help="Initial SG learning rate",
                           type=float, default=0.1, required=False)
    argparser.add_argument("--positive", help="Positive class",
                           type=str, default="../data/autos_motorcycles/positive", required=False)
    argparser.add_argument("--negative", help="Negative class",
                           type=str, default="../data/autos_motorcycles/negative", required=False)
    argparser.add_argument("--vocab", help="Vocabulary that can be features",
                           type=str, default="../data/autos_motorcycles/vocab", required=False)
    argparser.add_argument("--passes", help="Number of passes through train",
                           type=int, default=1, required=False)

    args = argparser.parse_args()
    train, test, vocab = read_dataset(args.positive, args.negative, args.vocab)

    print("Read in %i train and %i test" % (len(train), len(test)))

    # Initialize model
    lr = LogReg(len(vocab), args.lam, lambda x: 0.05) 

    print(args.eta)

    # Iterations
    iteration = 0
    for pp in xrange(1):
        random.shuffle(train)
        for ex in train:
            lr.sg_update(ex, iteration)
            if iteration % 5 == 1:
                train_lp, train_acc = lr.progress(train)
                ho_lp, ho_acc = lr.progress(test)
                print("Update %i\tTP %f\tHP %f\tTA %f\tHA %f" %
                      (iteration, train_lp, ho_lp, train_acc, ho_acc))
            iteration += 1

    print sorted(lr.w)
    midpoint = floor(len(lr.w)/2)
    worst = np.argsort(lr.w)[(midpoint-10):(midpoint+10)]
    print "\n" + "Worst Predictors"
    for elem in worst:
        print vocab[elem]
    autos = np.argsort(lr.w)[:15]
    print "\n" + "Automobiles"
    for elem in autos:
        print vocab[elem]
   
    print "\n" + "Motorcycles"
    motorcycles = np.argsort(lr.w)[-15:]
    for elem in motorcycles:
        print vocab[elem]
