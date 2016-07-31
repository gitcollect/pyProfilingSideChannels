from __future__ import print_function

__author__ = "Benny Pu, push.beni@gmail.com"
__docformat__ = 'restructedtext en'

import numpy
import theano
import theano.tensor as T

import future        # pip install future
import builtins      # pip install future
import six           # pip install six

from builtins import object,bytes,range,dict
from six import iteritems,with_metaclass
from io import open

from sklearn.decomposition import PCA, KernelPCA, RandomizedPCA, SparseCoder, DictionaryLearning, SparsePCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.cluster import FeatureAgglomeration
from sklearn.cross_decomposition import CCA
from sklearn import manifold

from utils import FilePath, load_pickle, save_pickle

def selectClassifier(x):
    return {
        'lda': (LDA(), True),

        'kld': (LDA(), True),

        'cca': (CCA(n_components=30), True),

        'pca': (PCA(n_components=10), False),

        'kpca': (KernelPCA(n_components=9, kernel='poly'), False),

        'lle': (manifold.LocallyLinearEmbedding(
            n_components=10, eigen_solver='auto', n_neighbors=10, method='modified'), False),  # `method` = 'ltsa', 'standard','modified','hessian'

        'tsne': (manifold.TSNE(n_components=8, random_state=2378, init='pca'), False),

        'mds': (manifold.MDS(n_components=2, random_state=2378, metric=False), False),
    }.get(x, (LDA(), True))


def dimReduct(X, y, reducemethod=None, picklefile=None, pathObj=None):
    # Dimensionality Reduction

    if reducemethod is None:
        print('NO REDUCTION')
        return X

    clf, supervised = selectClassifier(reducemethod)

    if supervised:
        if pathObj[u'reducefileL'] is None:
            if reducemethod is 'kld':
                clf_ = KernelPCA(n_components=200, kernel='poly', degree=3)
                # clf_ = DictionaryLearning(n_components=20)
                # clf_ = FeatureAgglomeration(n_clusters=30)
                # clf_ = manifold.TSNE(n_components=30,random_state=0, init='pca')
                clf_.fit(X)
                X_ = clf_.transform(X)
            else:
                clf_ = None
                X_ = X

            clf.fit(X_, y)
            save_pickle((clf_, clf), pathObj[u'reducefileS'])
        else:
            clf_, clf = load_pickle(pathObj[u'reducefileL'])
            if clf_ is None:
                X_ = X
            else:
                X_ = clf_.transform(X)

        return clf.transform(X_)

    else:
        if pathObj[u'reducefileL'] is None:
            clf.fit(X)
            save_pickle((None, clf), pathObj[u'reducefileS'])
        else:
            _, clf = load_pickle(pathObj[u'reducefileL'])
        return clf.transform(X)


if __name__ == '__main__':

    print('Nonthing.')
