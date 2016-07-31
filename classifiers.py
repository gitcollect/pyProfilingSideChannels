from __future__ import print_function

__author__ = "Benny Pu, push.beni@gmail.com"
__docformat__ = 'restructedtext en'


import os
import sys
import timeit

import numpy
import cPickle

import future
import builtins
import six
import csv

from abc import ABCMeta,abstractmethod
from builtins import object,bytes,range,dict
from six import iteritems,with_metaclass
from io import open

import theano
import theano.tensor as T


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from mlp import MLP
# from cnn import CNN
from utils import FilePath, load_data


# file extensions
known_extensions = {
    u'.csv': u'csv',
    u'.mat': u'matlab',
    u'.txt': u'ascii',
    u'.pkl': u'pickle'}


def formatFromExtension(filename):
    """Split the filename to get the extension.
    """
    root, ext = os.path.splitext(filename)
    if not ext:
        return None
    try:
        format = known_extensions[ext]
    except KeyError:
        format = None
    return format

def classifierFromClfname(clfname):
    if not clfname:
        return None
    try:
        clf = {
            u'svc': SVC(kernel = b'poly', C = 0.025),
            u'rf' : RandomForestClassifier(),
            u'abc': AdaBoostClassifier(),
            u'mlp': MLP([100]),
            # u'cnn': CNN()
        }.get(clfname)
    except KeyError:
        clf = None
    return clf


class FileHandler(with_metaclass(ABCMeta)):
    """Adapter for reading from a file and writing to a file.
    """
    # DEPRECATED after py3
    # __metaclass__ = ABCMeta

    # FIXME
    # csv file to be implemented
    # @abstractmethod
    # def _save_csv(self, fileobject, **kwargs):
    #    pass

    @abstractmethod
    def _save_pickle(self, fileobject, **kwargs):
        pass

    def _saveFileLike(self, fileobject, format=None, **kwargs):
        """Save obj to the file, format can be pickle, csv or txt.
        """
        format = 'pickle' if format is None else format
        save = getattr(self, "_save_%s" % format, None)
        if save is None:
            raise ValueError("Unknown format '%s' ." % format)
        save(fileobject, **kwargs)

    def saveFile(self, filename, format=None, **kwargs):
        if not format:
            format = formatFromExtension(filename)
        with open(filename, 'wb') as fileobject:
            self._saveFileLike(fileobject, format, **kwargs)


    @abstractmethod
    def _load_pickle(self, fileobject):
        pass

    def _loadFileLike(self, fileobject, format=None, **kwargs):
        """Load object to a given file"""
        format = 'pickle' if format is None else format
        load = getattr(self, "_load_%s" % format, None)
        if load is None:
            raise ValueError("Unknown format '%s'." % format)
        load(fileobject, **kwargs)

    def loadFile(self, filename, format=None, **kwargs):
        if not format:
            format = formatFromExtension(filename)
        with open(filename, 'rb') as fileobject:
             self._loadFileLike(fileobject, format, **kwargs)



class Dataset(FileHandler):
    """
    Dataset storing arrays for training.
    """
    def __init__(self, dsfile=None):
        if dsfile is None:
            self._data = None
            self._path = FilePath()
            self._params={
                u'datrange'       : None, # (0,1000)
                u'dim'            : None, # (200,1)
                u'pathObj'        : self._path,
                u'reducemethod'   : None, # u'kld'
                u'random'         : None,

                u'shift'          : 0,
                u'median'         : True,
                u'extend'         : False,
                u'hw'             : True
            }
        else:
            self.loadFile(dsfile)

    def __str__(self):
        s = "Dataset " + "traceset-" + str(self._params[u'datrange']) + " points-" + str(self._params[u'dim']) + " shift-" + str(self._params[u'shift']) + " rand-" + str(self._params[u'random']) + " reduc-" + str(self._params[u'reducemethod'])
        return s

    @property
    def params(self):
        """ parameters for of dataset. """
        return self._params

    @params.setter
    def params(self, valDict):
        self._params.update(valDict)
        # self._params = valDict

    @params.deleter
    def params(self):
        del self._params

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, filepathobj):
        self._path = filepathobj

    @path.deleter
    def path(self):
        del self._path


    def updateParams(self, **kwargs):
        if kwargs is not None:
            self._params.update(kwargs)
            print(self.params)
        else:
            print("No changed.")

    def _save_pickle(self, fileobject, **kwargs):
        if self._data is None:
            raise ValueError("No dataset to save. ")
        _tempDat = (self._data, self.params)
        cPickle.dump(_tempDat, fileobject)
        print("Datasets and params saved.")


    def _load_pickle(self, fileobject, **kwargs):
        _tempDat = cPickle.load(fileobject)
        self._data   = _tempDat[0]
        self.params  = _tempDat[1]
        print("Datasets and params loaded.")

    def getTrain(self):
        data  = self._data[0][0].get_value()
        label = self._data[0][1].eval()
        return (data, label)

    def getTest(self):
        data  = numpy.concatenate((self._data[1][0].get_value(),self._data[2][0].get_value()),0)
        label = numpy.concatenate((self._data[1][1].eval(),self._data[2][1].eval()),0)
        return (data, label)

    def getTheanoTrain(self):
        return (self._data[0], self._data[1])

    def getTheanoTest(self):
        return (self._data[2][0], self._data[2][1])


    def construct(self, **kwargs):
        if kwargs is not None:
            self.params.update(kwargs)
        # self.updateParams(pathObj=self._path)
        self._data = load_data(**self.params)



class Classifier(FileHandler):
    """
    A classifier learning model.
    """
    def __init__(self, clffile=None):
        if clffile is None:
            self._classifier    = None
            self._clfname       = None
        else:
            self.loadFile(clffile)

    def __str__(self):
        return self._clfname


    def _save_pickle(self, fileobject):
        if self._classifier is None:
            raise ValueError("No classifier to save. ")
        print("...saving")
        cPickle.dump(self._classifier, fileobject)
        print("done.")

    def _load_pickle(cls, fileobject):
        print("...loading")
        self._classifier = cPickle.load(fileobject)
        print("done.")

    def setClassifier(self, clfname):
        self._classifier = classifierFromClfname(clfname)
        self._clfname    = clfname

    def trainModel(self, ds):
        if self._clfname in (u'cnn', u'mlp'):
            X, Y = ds.getTheanoTrain()
            print("The shape of trainning set: (rows: %i, cols: %i) for data and %i for label"
                    % (X[0].get_value().shape[0], X[0].get_value().shape[1], X[1].eval().shape[0]))
        else:
            X, Y = ds.getTrain()
            print("The shape of trainning set: (rows: %i, cols: %i) for data and %i for label"
                    % (X.shape[0], X.shape[1], Y.shape[0]))
            # print(type(X), type(Y))

        print("...training")
        if self._classifier is None:
            raise ValueError("No clasifier exist.")

        self._classifier.fit(X, Y)
        print("Trainning process finished.")

    def predict(self, ds):
        if self._clfname in (u'cnn', u'mlp'):
            X, Y = ds.getTheanoTest()
            print("The shape of testing set: (rows: %i, cols: %i) for data and %i for label"
                    % (X.get_value().shape[0], X.get_value().shape[1], Y.eval().shape[0]))
        else:
            X, Y = ds.getTest()
            print("The shape of testing set: (rows: %i, cols: %i) for data and %i for label"
                    % (X.shape[0], X.shape[1], Y.shape[0]))

        if self._classifier is None:
            raise ValueError("No clasifier exist.")

        pred_score = self._classifier.score(X, Y)
        pred       = self._classifier.predict(X)

        print ('Predicted errors: \n', T.neq(pred, Y if type(Y) is numpy.ndarray else Y.eval()).eval())
        print ('Accuracy: \n', pred_score)







if __name__ == '__main__':

    ds    = Dataset()
    param = {u'datrange':[0,1000], u'dim':[50,1]}

    ds.params = param

    print(ds.params)
    print("Feat file -> ", ds.path[u'featfileL'], "\nCorr file -> ", ds.path[u'corrfileL'], "\nReduc file -> ", ds.path[u'reducefileL'])

    ds.path[u'featfileL'] = ds.path[u'featfileS']

    ds.loadFile(str(ds)+".pkl")

    # ds.construct()
    # ds.saveFile(str(ds)+".pkl")

    mdl = Classifier()
    mdl.setClassifier('mlp')

    mdl.trainModel(ds)
    # mdl.predict(ds)

    #############################################
    #############################################

    # tst = (u'feat_list.pkl', u'clf.pkl')
    # # datasets_rand = load_data(start = 4000, end = 5000, median = True, dim = 200, k = 1, reducemethod = dimReductMethod, test = tst, rebuild = 'gauss', shift = 5)
    # # util.save_pickle(datasets_rand, 'D:/GoogleDrive/dats/datasets_rand.pkl')


    # datasets_ = load_data(start = 2000, end = 3000, median = True, dim = 200, k = 1, reducemethod = dimReductMethod, test = tst, rebuild = True, shift= 5, random='uniform')

    # util.save_pickle(datasets_, 'D:/GoogleDrive/dats/datasets_.pkl')


    # evaluate_clf(clfname = 'svc', datasets = datasets, datasets_ = datasets_, picklefile = None)
    # evaluate_clf(clfname = 'svc', datasets = datasets, datasets_ = None, picklefile = 'classifier.pkl')
