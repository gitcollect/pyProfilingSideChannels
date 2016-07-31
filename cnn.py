"""
NOTES:

 - location-specific gain and bias parameters

 - pooling by average or by max:
   actually, in trace recog, `max pooling` is better than `sum` or `average pooling`

 - using `logistic regression` or `RBF network`

"""

from __future__ import print_function

__author__ = "Benny Pu, push.beni@gmail.com"


import os
import sys
import timeit

import numpy
import future        # pip install future
import builtins      # pip install future
import six           # pip install six

from builtins import object, bytes, range, dict
from six import iteritems, with_metaclass

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d

from mlp import HiddenLayer, LogisticRegression


class ConvPoolLayer(object):
    """Pool Layer of a convolutional network.
    """

    def __init__(self, rng, input, image_shape, filter_shape,  poolsize):
        print ("image_shape: ", image_shape, "\nfilter_shape: ", filter_shape, : "\npool_size: ", poolsize)
        assert image_shape[1] == filter_shape[1]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True,
            mode='max'  # 'max','sum','average_inc_pad','average_exc_pad'
        )

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input


class CNNModel(object):

    def __init__(self, rng, input, batch_size,
                 nkerns=[], hidunits=[], poolsize=[], n_in, n_out):

        self.cplayers = [None] * len(nkerns)
        self.hidlayers = [None] * len(hidunits)
        self.points = [None] * len(nkerns)

        for layerIdx in list(range(len(nkerns))):
            if layerIdx < 1:
                self.points[layerIdx] = (n_in[0], n_in[1])
            else:
                
                self.points[layerIdx] = ((self.points[layerIdx - 1][0] - nkerns[layerIdx - 1][2] + 1) // poolsize[0],
                                         (self.points[layerIdx - 1][1] - nkerns[layerIdx - 1][3] + 1) // poolsize[1])
            self.cplayers[layerIdx] = ConvPoolLayer(
                rng=rng,
                input=input.reshape((batch_size, 1, n_in[0], n_in[
                                    1])) if layerIdx < 1 else self.cplayers[layerIdx - 1].output,
                filter_shape=nkerns[layerIdx],
                image_shape=(batch_size, 1 if layerIdx < 1 else nkerns[
                             layerIdx - 1][0], self.points[layerIdx][0], self.points[layerIdx][1]),
                poolsize=poolsize,
            )

        for layerIdx in list(range(len(hidunits))):

            self.hidlayers[layerIdx] = HiddenLayer(
                rng=rng,
                input=self.cplayers[len(nkerns) - 1].output.flatten(
                    2) if layerIdx < 1 else self.hidlayers[layerIdx - 1].output,
                n_in=(nkerns[len(nkerns) - 1][0] * self.points[len(nkerns) - 1][0] *
                      self.points[len(nkerns) - 1][1])if layerIdx < 1 else hidunits[layerIdx - 1],
                n_out=hidunits[layerIdx],
                activation=T.tanh
            )

        # Classify the values of the fully-connected sigmoidal layer
        self.lgsrlayer = LogisticRegression(input=self.hidlayers[len(
            hidunits) - 1].output, n_in=hidunits[len(hidunits) - 1], n_out=n_out)

        # Followinga are instance methods, cannot be `pickled`!!
        # self.params = self.layer3.params + self.layer2.params + self.layer1.params + self.layer0.params
        # self.negLL  = self.layer3.negative_log_likelihood
        self.input = input

class CNN(object):
    def __init__(self, nkerns=[], hidunits=[], poolsize=[]):
        self._classifier = None
        self._nkerns     = nkerns
        self._hidunit    = hidunits
        self._poolsize   = poolsize

    def fit(self, trainset, validset, n_epochs=300, learning_rate=0.05, batch_size=10):
        ##############
        ## SETTINGS ##
        ##############
        train_set_x, train_set_y = trainset
        valid_set_x, valid_set_y = validset

        tr_samples      =  train_set_x.get_value(borrow=True).shape[0]
        va_samples      =  valid_set_x.get_value(borrow=True).shape[0]

        n_features      =  train_set_x.get_value(borrow=True).shape[1]
        n_labels        =  len(numpy.unique(train_set_y.eval()))

        assert tr_samples == train_set_y.eval().shape[0]

        n_train_batches = tr_samples // batch_size
        n_valid_batches = va_samples // batch_size

        print('# of train batches: ', n_train_batches)
        print('# of valid batches: ', n_valid_batches)

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print('... building the model')
        # allocate symbolic variables for the data
        index = T.lscalar()     # index to a [mini]batch
        x     = T.matrix('x')
        y     = T.ivector('y')  # the labels are presented as 1D vector of
                                # [int] labels
        rng = numpy.random.RandomState(1234)

        self._classifier = CNNModel(
            rng          = rng,
            input        = x,
            batch_size   = batch_size,

            nkerns       = self._nkerns,
            poolsize     = self._poolsize,
            hidunits     = self._hidunit,
            n_in         = n_features,
            n_out        = n_labels
        )

        params = self._classifier.lgsrlayer.params

        for indexlayer in list(range(len(self._classifier.cplayers))):
            params = params + self._classifier.cplayers[indexlayer].params
        for indexlayer in list(range(len(self._classifier.hidlayers))):
            params = params + self._classifier.hidlayers[indexlayer].params

        cost = self._classifier.lgsrlayer.negative_log_likelihood(y)
        grads = T.grad(cost, params)

        # updating rule: SGD
        updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(params, grads)
        ]

        validate_model = theano.function(
            inputs  = [index],
            outputs = self._classifier.lgsrlayer.error(y),
            givens  = {
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        train_model = theano.function(
            inputs  = [index],
            outputs = cost,
            updates = updates,
            givens  = {
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        ###############
        # TRAIN MODEL #
        ###############

        print('... training')
        # early-stopping parameters
        patience = 10000
        patience_increase = 2
        improvement_threshold = 0.995

        # considered significant
        validation_frequency = min(n_train_batches, patience // 2)

        best_validation_loss = numpy.inf
        best_iter  = 0
        test_score = 0.
        best_clf   = None
        start_time = timeit.default_timer()

        epoch = 0
        done_looping = False

        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            learning_rate = learning_rate * 0.9
            for minibatch_index in range(n_train_batches):

                iter = (epoch - 1) * n_train_batches + minibatch_index

                if iter % 100 == 0:
                    print('training @ iter = ', iter)
                cost_ij = train_model(minibatch_index)

                if (iter + 1) % validation_frequency == 0:

                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(
                        i) for i in range(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)

                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches,
                           this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        # improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter
                        best_clf  = self._classifier


                if patience <= iter:
                    done_looping = True
                    break

        end_time = timeit.default_timer()
        print('Optimization complete.')
        print('Best validation score of %f %% obtained at iteration %i, '

              'with test performance %f %%' %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print(('The code for file ' +
               os.path.split(__file__)[1] +
               ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

    def _pred(self, test_set_x):
        """ Predict the new coming samples on testing sets.
        """
        test_set_x_val         = test_set_x.get_value()

        perdict_model = theano.function(
            inputs  = [self._classifier.input],
            outputs = (self._classifier.lgsrlayer.y_pred, self._classifier.lgsrlayer.p_y_given_x)
        )

        predict_values, predict_probs = perdict_model(test_set_x_val)
        return (predict_values, predict_probs)

    def predict(self, test_set_x):
        rval = self._pred(test_set_x)
        return rval

    def score(self, test_set_x, test_set_y):
        rval  = self._pred(test_set_x)
        error = sum(T.neq(rval, test_set_y.eval()).eval())
        return 1-error



if __name__ == '__main__':
    print('Nothing.')

    # #datasets = util.load_pickle('D:\\GoogleDrive\\dats\\datasets.pkl')

    # ## reload datasets
    # datasets = load_data(start = 0000, end = 5000, median=True, dim = 10, reduce = False,
    #         )
    # #util.save_pickle(datasets, 'D:/GoogleDrive/dats/datasets.pkl')

    # #datasets_ = load_data(start = 0000, end = 5000, median=True, dim = 3, reduce = False,
    # #    rebuild = True, shift = 10)

    # batch_s = 20

    # evaluate_lenet( learning_rate = 0.01,
    #                      n_epochs = 500,
    #                    batch_size = batch_s,
    #                          n_in = (16,10),
    #                       n_out = 256,
    #                    datasets = datasets,
    #                      datasets_ = None
    #                 )
    # predict(test_set = datasets[2], start = 0, batch_size = batch_s, n_in = (16,10))
