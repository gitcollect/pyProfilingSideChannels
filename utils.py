from __future__ import print_function

__author__ = "Benny Pu, push.beni@gmail.com"

__docformat__ = 'restructedtext en'


import os
import time
import cPickle
import csv

import numpy
import future        # pip install future
import builtins      # pip install future
import six           # pip install six

from six import iteritems,with_metaclass
from builtins import object,dict,range
from io import open

import theano
import theano.tensor as T
from scipy.stats.stats import pearsonr
from sklearn import preprocessing
# import matplotlib.pyplot as plt

import aes

class ClassProperty(type):
    @property
    def tracefile(self):
        return self._tracefile

    @property
    def indexfile(self):
        return self._indexfile




class FilePath(object):

    def __init__(self, **kwargs):
        self._tracefile   = u'D:/works/aes-rsm/DPA_contestv4_rsm_00000/DPA_contestv4_rsm/00000'
        self._indexfile   = u'D:/works/aes-rsm'

        self._corrfileL   = None
        self._corrfileS   = u'D:/GoogleDrive/dats/pear_sinHW.pkl'

        self._featfileL   = None
        self._featfileS   = u'featfile.pkl'

        self._reducefileL = None
        self._reducefileS = u'reducefile.pkl'

        self._lenth       = 8

        if kwargs is not None:
            for (key, val) in iteritems(kwargs):
                attr = getattr(self, u"_%s" % key, None)
                self._lenth += 1
                if attr is None:
                    setattr(self, attr, val)
                else:
                    attr = val

    def __getitem__(self, file):
        return getattr(self, u"_%s" % file, None)

    def __setitem__(self, file, path):
        setattr(self, u"_%s" % file, path)
        self._lenth += 1

    def __delitem__(self, file):
        attr =  getattr(self, u"_%s" % file, None)
        if attr is not None:
            del attr
            self._lenth -= 1
        else:
                print("No filepath " + file + " found. ")

    def __len__(self):
        return self._lenth






# read the data matrix from `trc` files
# :type: 2d-array
def readBinary(start, end, pathObj):
    start_time = time.time()
    rawdat = numpy.zeros((end-start,435002), dtype='float32')

    print("... reading from binary files.\n")
    for i in list(range(end-start)):
        rawdat[i] = numpy.fromfile(
            pathObj[u'tracefile']+'\\Z1Trace'+format(start+i,'05d')+'.trc',
            dtype = 'int8',
            count = -1
        )[357:]
        #print("reading... "+str(i))

    end_time = time.time()

    print ("time using: ")
    print (end_time-start_time)
    return rawdat



# read offset,plain/ciphertext,key,etc., from `index.txt` file
# :type: 2d-array
def readIndex(start, end, pathObj):
    with open(pathObj[u'indexfile']+'/index.txt') as f:
        datlist = f.readlines()
    index_arr = numpy.zeros((len(datlist),6), dtype='|S64')
    for i in list(range(len(datlist))):
        index_arr[i] = numpy.asarray(datlist[i].split())
    return index_arr[start:end]



# left rotate (cyclic-left-shift)
def shiftLeft(seq, n):
    n = n % len(seq)
    return seq[n:] + seq[:n]

def shiftRightForArray(seq, n):
    n = (len(seq)-n) % len(seq)
    return numpy.concatenate((seq[:,n:], seq[:,:n]), axis = 1)


# get the real masking order for each trace
# :type: 2d-array
def getMaskVector(lis):
    mask = [0x00,0x0f,0x36,0x39,0x53,0x5c,0x65,0x6a,0x95,0x9a,0xa3,0xac,0xc6,0xc9,0xf0,0xff]
    maskVec = numpy.zeros((len(lis),16), dtype='int')
    for i in list(range(len(lis))):
        offset = int(lis[i,3],16)
        maskVec[i] = numpy.asarray(shiftLeft(mask,offset),dtype='int')
    return maskVec


def getMedVecs(lis, mvec, hw = True):
    ainVec = numpy.zeros((len(lis),16),dtype = 'int')
    sinVec = numpy.zeros((len(lis),16),dtype = 'int')
    soutVec = numpy.zeros((len(lis),16),dtype = 'int')

    medians = [aes.getMedians(lis[i,1],lis[i,0]) for i in list(range(len(lis)))]
    masklis = [shiftRightForArray(mvec,1), mvec, shiftRightForArray(mvec,1)]

    for i in list(range(len(lis))):
        if hw is True:
            ainLis = [bin(medians[i][0][j]^masklis[0][i,j]).count('1') for j in list(range(16))]
            sinLis = [bin(medians[i][1][j]^masklis[1][i,j]).count('1') for j in list(range(16))]
            soutLis = [bin(medians[i][2][j]^masklis[2][i,j]).count('1') for j in list(range(16))]
        else:
            ainLis = [medians[i][0][j]^masklis[0][i,j] for j in list(range(16))]
            sinLis = [medians[i][1][j]^masklis[1][i,j] for j in list(range(16))]
            soutLis = [medians[i][2][j]^masklis[2][i,j] for j in list(range(16))]


        ainVec[i] = numpy.array(ainLis, dtype = 'int')
        sinVec[i] = numpy.array(sinLis[0::2]+sinLis[1::2], dtype='int')
        soutVec[i] = numpy.array(soutLis[0::2]+soutLis[1::2], dtype='int')

    return (ainVec, sinVec, soutVec)



def cpaAtSin(testMedians, lis):
    mvec = getMaskVector(lis)
    guessMedians = numpy.zeros((256,len(lis)),'int')
    print('... guessing values')
    for key in list(range(256)):
        guessMedians[key] = numpy.array([bin(aes.getMedians(lis[i,1], key, test=True)[1][0]^mvec[i,0]).count('1') for i in list(range(len(lis)))], dtype='int')
        print('key: ' + str(key) + ' finished...')

    result = numpy.zeros(256,'float32')
    print('... correlating')
    for key in list(range(256)):
        result[key] = pearsonr(guessMedians[key], testMedians)[0]
        print('key: ' + str(key) + ' finished...')
    print('done.')
    return result


# offset values
def getLabel(lis):
    label = numpy.zeros(len(lis),dtype = 'int')
    for i in list(range(len(lis))):
        label[i] = int(lis[i,3],16)
    return label



def getLabelMedians(lis, mvec, bytesIdx, hw=True):
    if hw:
        medians = [bin(aes.getMedians(lis[i,1],lis[i,0])[1][bytesIdx]^mvec[i,bytesIdx]).count('1')  for i in list(range(len(lis)))]
    else:
        medians = [aes.getMedians(lis[i,1],lis[i,0])[1][bytesIdx]^mvec[i,bytesIdx]  for i in list(range(len(lis)))]

    return numpy.array(medians,dtype='int')




def getPearson(dat, mvec):
    print("two arrays to do pearson-r-correlation", dat.shape, mvec.shape)
    if dat.shape[0] != mvec.shape[0]:
        raise Exception("Cannot do correlation: two vectors must have same dim in len()")
    res = numpy.zeros((dat.shape[1], mvec.shape[1]), dtype='float32')
    print('... correlation evaluating')
    for j in list(range(mvec.shape[1])):
        for  i in list(range(dat.shape[1])):
            res[i,j] = pearsonr(dat[:,i], mvec[:,j])[0]
        print(('... # col %d finished. ') % j)
    print('done.')
    return res



def selectDistribution(random, shift, mean=0):
    return {
        'gauss': numpy.random.normal(mean,shift),
        'uniform': numpy.random.randint(-shift, shift)
    }.get(random, shift)


def shiftTraces(dat, shift, random=u'gauss'):
    dat_ = numpy.zeros((dat.shape[0],dat.shape[1]),dat.dtype)
    for i in list(range(dat.shape[0])):
        if random is None:
            offset = shift
        else:
            offset = int(selectDistribution(random, shift))

        if offset > 0:
            dat_[i] = numpy.append(numpy.zeros(offset), dat[i][:len(dat[i])-offset])
        else:
            dat_[i] = numpy.append(dat[i][-offset:], numpy.zeros(-offset))

    return dat_


def save_csv(obj, filepath):
    print(obj.shape)
    with open(filepath, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(obj)


def save_pickle(obj, filepath):
    with open(filepath,'wb') as filobj:
        cPickle.dump(obj, filobj)


def load_pickle(filepath):
    with open(filepath, 'rb') as filobj:
        return cPickle.load(filobj)


def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables
        `borrow` is `False` means make a [deepcopy] of object,
         otherwise the `shared` objects can be updated.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y.flatten()
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')


def featureExtract(dat, dim, mvec, pathObj, idx, pos=u'pearson', median=True, discrete=True, bound=0):
    ##########################
    ##  FEATURE EXTRACTION  ##
    ##########################
    if pos == u'pearson':
        if pathObj[u'corrfileL'] is None:
            res = getPearson(dat=dat, mvec=mvec).reshape(dat.shape[1], mvec.shape[1])
            save_pickle(res, pathObj[u'corrfileS'])
        else:
            if idx is None:
                res = abs(load_pickle(pathObj[u'corrfileL']).reshape(dat.shape[1], mvec.shape[1]))
            else:
                res = abs(load_pickle(pathObj[u'corrfileL'])[:,idx].reshape(dat.shape[1], 1))

    elif pos == u'diff':
        if median:
            inds = mvec[:,idx].argsort()
            dat_sort = dat[inds]
            res = abs(dat_sort[len(dat_sort)-1]-dat_sort[0]).reshape(dat_sort.shape[1], 1)
        else:
            raise Exception('Non-median values cannot use `diff` profiling! Please use `pos=\'pearson\'`! ')

    print ('shape of correlation result: ', res.shape)
    print ('shape of feat list: ', dim)

    pos = numpy.zeros(dim, dtype='int')

    if discrete is True:
        for i in list(range(dim[0])):
            pos[i] = numpy.argmax(res, 0)
            toDel = numpy.where(pos[i] > bound)[0] if bound > 0 else []
            while len(toDel) > 0:
                for j in toDel:
                    res[pos[i, j], j] = 0
                pos[i] = argmax(res, 0)
                toDel = numpy.where(pos[i] > bound)[0] if bound > 0 else []
            for j in list(range(pos.shape[1])):
                res[pos[i, j], j] = -1 - res[pos[i, j], j]
    else:
        mid = dim[0]//2
        pos[mid] = argmax(res, 0)
        toDel = numpy.where(pos[i] > bound)[0] if bound > 0 else []
        while(len(toDel) > 0):
            for j in toDel:
                res[pos[mid, j], j] = 0
            pos[mid] = argmax(res, 0)
            toDel = numpy.where(pos[i] > bound)[0] if bound > 0 else []

        # pos[mid].sort()
        for i in list(range(dim[0])):
            if i == mid:
                continue
            pos[i] = pos[mid] + (i-mid)

    pos = pos.reshape(1, dim[1]*dim[0])
    pos.sort()
    print('feature extraction finished.')
    return pos



def load_data(datrange, dim, pathObj, shift, reducemethod=None, random='gauss', median=True, hw=True, extend =False):
    ###############
    ## LOAD DATA ##
    ###############
    print('... loading traces:  index from %d to %d' % (datrange[0], datrange[1]))

    dat_set = readBinary(start=datrange[0], end=datrange[1], pathObj=pathObj)
    dat_idx = readIndex(start=datrange[0], end=datrange[1], pathObj=pathObj)

    print('... spliting datasets into three parts')
    idx = numpy.random.permutation(len(dat_set))
    tr_idx, va_idx, te_idx = idx[:int(len(
        idx) * 0.8)], idx[int(len(idx) * 0.8):int(len(idx) * 0.9)], idx[int(len(idx) * 0.9):]

    ##############
    ## SHIFTING ##
    ##############
    if shift > 0:
        print('... shifting traces')
        if extend:
            print('... extending')
            dat_set_ = numpy.zeros((dat_set.shape[0]*(2*shift+1),dat_set.shape[1]), dat_set.dtype)
            for i in list(range(2*shift+1)):
                if i == shift:
                    dat_set_[i*dat_set.shape[0]:(i+1)*dat_set.shape[0],:] = dat_set
                else:
                    dat_set_[i*dat_set.shape[0]:(i+1)*dat_set.shape[0],:] = shiftTraces(dat=dat_set, shift=shift-i, random=random)
            dat_set = dat_set_
        else:
            dat_set = shiftTraces(dat=dat_set, shift=shift, random=random)

    print('shape of trace set: ', dat_set.shape )


    maskVec = getMaskVector(dat_idx)
    medVec  = getMedVecs(dat_idx, maskVec, hw=hw)[1] # sbox_in:'1', sbox_out:2', addroundKey_2nd:'0'

    ################
    ## BYTE-INDEX ##
    ################
    idx = 0 if dim[1] == 1 else None

    ##############
    ## FEATURES ##
    ##############
    if pathObj[u'featfileL'] is not None:
        feat_list = load_pickle(pathObj[u'featfileL'])
    else:
        print('... feature extracting')
        feat_list = featureExtract(dat=dat_set[tr_idx,:], dim=dim, pathObj=pathObj, idx=idx, mvec=medVec[tr_idx,:] if median else maskVec[tr_idx,:], median=median)
        save_pickle(feat_list, pathObj[u'featfileS'])
    print('feat list: \n', feat_list)

    print('... getting labels')
    _label_list = getLabelMedians(lis=dat_idx, mvec=maskVec, bytesIdx=2*idx, hw=hw) if median else getLabel(dat_idx)
    label_list  = numpy.concatenate((_label_list,)*(shift*2+1),0) if extend else _label_list
    print('the shape of label_list : ', label_list.shape )

    #############
    ## SCALING ##
    #############
    # dat_set = preprocessing.scale(dat_set)


    ###############
    ## REDUCTION ##
    ###############
    print('... dimensionality reduction')
    if reducemethod is not None:
        print('REDUCTION %s USED', reducemethod)
        feat_set_tr_idx = reduc.dimReduct(dat_set[tr_idx,:][:,feat_list[0]], label_list[tr_idx], reducemethod, pathObj)

        pathObj[u'reducefileL'] = pathObj[u'reducefileS']
        feat_set_va_idx = reduc.dimReduct(dat_set[va_idx,:][:,feat_list[0]], None, reducemethod, pathObj)
        feat_set_te_idx = reduc.dimReduct(dat_set[te_idx,:][:,feat_list[0]], None, reducemethod, pathObj)

        train_set = (feat_set_tr_idx, label_list[tr_idx])
        valid_set = (feat_set_va_idx, label_list[va_idx])
        test_set = (feat_set_te_idx, label_list[te_idx])

    else:
        print('NO REDUCTION USED')
        feat_set = dat_set[:,feat_list[0]]
        train_set = (feat_set[tr_idx], label_list[tr_idx])
        valid_set = (feat_set[va_idx], label_list[va_idx])
        test_set = (feat_set[te_idx], label_list[te_idx])


    # Export to a csv file
    # save_csv(numpy.concatenate((feat_set, label_list.reshape(len(feat_set),1)),1), u'D:/GoogleDrive/dats/ds_' + str(dim) + u'dim.csv')

    ##############
    ## WRAPPING ##
    ##############
    print('wrapping datasets by theano')
    test_set_x, test_set_y   = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    print('datasets loaded.')
    return rval




 # DEPRECATED
def rebuild_set(dat, shift):
    tr, va, te = dat
    tr_set = (shiftTraces(dat=tr[0].get_value(),shift=shift), tr[1].eval())
    te_set = (shiftTraces(dat=te[0].get_value(),shift=shift), te[1].eval())
    va_set = (shiftTraces(dat=va[0].get_value(),shift=shift), va[1].eval())

    test_set_x, test_set_y = shared_dataset(te_set)
    valid_set_x, valid_set_y = shared_dataset(va_set)
    train_set_x, train_set_y = shared_dataset(tr_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    print('... rebuilding done.')

    return rval


if __name__ == '__main__':
    print('Nothing.')
    # start = 0000
    # end = 5000

    # dat = readBinary(start,end)
    # idx = readIndex(start,end)
    # mask = getMaskVector(idx)

    # med= getMedVecs(idx,mask,hw=True)[1]

    # inds = med[:,0].argsort()
    # datSorted = dat[inds]
    # cur = [0,0]
    # sum_ = numpy.zeros(dat.shape[1], 'int')
    # keyval = numpy.zeros((256,dat.shape[1]),'int')

    # for i in list(range(len(datSorted))):
    #     if med[inds[i],0]==cur[0]:
    #         sum_ += datSorted[i,:]
    #         cur[1] += 1
    #     else:
    #         if cur[1] > 0:
    #             keyval[cur[0]] = sum_/cur[1]
    #         cur[0] += 1
    #         cur[1] = 0
    #         sum_ = numpy.zeros(dat.shape[1],'int')
    #     if i == len(datSorted)-1 :
    #         keyval[cur[0]] = sum_/cur[1]
    #         cur[1] = 0
    #         cur[0] = 0


    # mediansLabel = getLabelMedians(idx,mask)

    # print(med, mediansLabel)


    # pear_sin  = getPearson(dat, med)

