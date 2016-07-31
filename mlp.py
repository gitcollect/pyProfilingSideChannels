
from __future__ import print_function

__author__ = "Benny Pu, push.beni@gmail.com"

__docformat__ = 'restructedtext en'


import os
import sys
import timeit

import numpy
import future        # pip install future
import builtins      # pip install future
import six           # pip install six

from builtins import object,bytes,range,dict
from six import iteritems,with_metaclass

import theano
import theano.tensor as T
from theano.tensor.signal import pool


class PoolLayer(object):
    def __init__(self, input, n_in = 48, poolsize = (1,2)):

        pooled_out = pool.pool_2d(
            input=input,
            ds   =poolsize,
            ignore_border=True,
            mode ='max'  # 'max','sum','average_inc_pad','average_exc_pad'
        )
        #self.output= [T.max(input[i*poolsize:i*poolsize+poolsize]) for i in range(n_in//poolsize)]
        self.input = input
        self.output = pooled_out
        self.n_out = n_in//poolsize[1]


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).
        """
        self.input = input
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


class LogisticRegression(object):
    """Multi-class Logistic Regression Class
        --i.e. softmax func...
    """
    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression
        """
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.
        """
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class MLPModel(object):

    def __init__(self, rng, input, n_in, n_hidden, n_out):

        # FIXME
        # No need to use pool layer in this traditional neural networks.
        # self.poolLayer = PoolLayer(input = input, n_in = n_in , poolsize = (1,2))
        self.layers       = len(n_hidden)
        self.hiddenLayers = [None]*self.layers

        for layer in list(range(self.layers)):
            self.hiddenLayers[layer] = HiddenLayer(
                rng             = rng,
                input           = input if layer < 1 else self.hiddenLayers[layer-1].output,
                n_in            = n_in if layer < 1 else n_hidden[layer-1],
                n_out           = n_hidden[layer],
                activation      = T.tanh
                )


        self.logRegressionLayer = LogisticRegression(
            input               = input if self.layers is 0 else self.hiddenLayers[self.layers-1].output,
            n_in                = n_in if self.layers is 0 else n_hidden[self.layers-1],
            n_out               = n_out
        )

        self.L1 = (
            sum( abs(self.hiddenLayers[i].W).sum() for i in list(range(self.layers)) )
            + abs(self.logRegressionLayer.W).sum()
        )

        self.L2_sqr = (
            sum( (self.hiddenLayers[i].W ** 2).sum() for i in list(range(self.layers)) )
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # keep track of model input
        self.input = input


class MLP(object):

    def __init__(self, n_hidden=None):
        self._classifier  = None
        self._n_hidden    = n_hidden


    def fit(self, trainset, validset, n_epochs=300, learning_rate=0.05, batch_size=10, L1_reg=0.00, L2_reg=0.001):
        ##############
        ## SETTINGS ##
        ##############

        train_set_x, train_set_y = trainset
        valid_set_x, valid_set_y = validset

        tr_samples      =  train_set_x.get_value(borrow=True).shape[0]
        va_samples      =  valid_set_x.get_value(borrow=True).shape[0]

        n_features      =  train_set_x.get_value(borrow=True).shape[1]
        n_labels        =  len(numpy.unique(train_set_y.eval()))

        n_hidneurons    =  tr_samples // (2 * (n_features + n_labels)) if self._n_hidden is None else self._n_hidden

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

        self._classifier = MLPModel(
            rng      = rng,
            input    = x,
            n_in     = n_features,
            n_hidden = n_hidneurons,
            n_out    = n_labels
        )

        cost = (
            self._classifier.logRegressionLayer.negative_log_likelihood(y)
            + L1_reg * self._classifier.L1
            + L2_reg * self._classifier.L2_sqr
        )

        params = self._classifier.logRegressionLayer.params

        for indexlayer in list(range(self._classifier.layers)):
            params = params + self._classifier.hiddenLayers[indexlayer].params

        gparams = [T.grad(cost, param) for param in params]

        # (variable, update expression) pairs
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(params, gparams)
        ]

        validate_model = theano.function(
            inputs     = [index],
            outputs    = self._classifier.logRegressionLayer.errors(y),
            givens     = {
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]
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

        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(n_train_batches, patience // 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        best_validation_loss = numpy.inf
        best_iter  = 0
        best_clf   = None
        test_score = 0.
        start_time = timeit.default_timer()

        epoch = 0
        done_looping = False

        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            learning_rate = learning_rate * 0.9
            for minibatch_index in list(range(n_train_batches)):

                minibatch_avg_cost = train_model(minibatch_index)
                # iteration number
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i
                                         in list(range(n_valid_batches))]
                    this_validation_loss = numpy.mean(validation_losses)

                    print(
                        'epoch %i, minibatch %i/%i, validation error %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            this_validation_loss * 100.
                        )
                    )

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if (
                            this_validation_loss < best_validation_loss *
                            improvement_threshold
                        ):
                            patience = max(patience, iter * patience_increase)

                        best_validation_loss = this_validation_loss
                        best_iter = iter
                        best_clf  = self._classifier


                if patience <= iter:
                    done_looping = True
                    break

        end_time = timeit.default_timer()
        print(('Optimization complete. Best validation score of %f %% '
               'obtained at iteration %i.\n') %
              (best_validation_loss * 100., best_iter + 1))

        print(('The code for file ' +
               os.path.split(__file__)[1] +
               ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

        # update the classifier
        self._classifier = best_clf


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
    print("Nothing.")

    # dimReductMethod = 'kld'

    # tst = None

    # datasets = load_data(start = 0000, end = 1000, median = True, dim = 500, k = 1, hw = True, reducemethod = dimReductMethod, test=tst)
    # util.save_pickle(datasets, 'D:/GoogleDrive/dats/datasets.pkl')

    # tst = 'all'

    # datasets_rand = load_data(start = 5000, end = 6000, median = True, dim = 500, k = 1, hw = True, reducemethod = dimReductMethod, test = tst, rebuild = True, shift = 5)
    # util.save_pickle(datasets_rand, 'D:/GoogleDrive/dats/datasets_rand.pkl')

    # datasets_ = load_data(start = 3000, end = 4000, median = True, dim = 500, k = 1, hw = True, reducemethod = dimReductMethod, test=tst, rebuild = True, shift = 5, random = None)
    # util.save_pickle(datasets_, 'D:/GoogleDrive/dats/datasets_.pkl')

    # # datasets = util.load_pickle('D:/GoogleDrive/dats/datasets.pkl')
    # # datasets_rand = util.load_pickle('D:/GoogleDrive/dats/datasets_rand.pkl')
    # # datasets_ = util.load_pickle('D:/GoogleDrive/dats/datasets_.pkl')

    # batch_s = 1
    # n_features = datasets[0][0].get_value().shape[1]
    # print('input features: ', n_features)

    # evaluate_mlp(     learning_rate = 0.0001,
    #                          L1_reg = 0.000,
    #                          L2_reg = 0.00,
    #                          # L2_reg = 0.005,
    #                        n_epochs = 500,
    #                        datasets = datasets,
    #                        datasets_= datasets_rand,
    #                      batch_size = batch_s,
    #                        n_hidden = (500,10),
    #                        n_in     = n_features,
    #                        n_out    = 9
    #                 )

    # pred_ = predict(test_set = datasets_rand[2], start = 0, batch_size = 20*batch_s)

    #########
    ## PREDICT on test samples, then do CPA to get the secret key byte

    # dataTest   = load_test(start = 9000, end = 9500, rebuild = False, shift = 10, median = True, reduc = 'lda')
    # listTest   = util.readIndex(9000,9500)

    # size = 50
    # pred_val = predict(test_set = dataTest, start = 0, batch_size = size)
    # print (pred_val)

    # rslt = util.cpaAtSin(numpy.array(pred_val,'int'),listTest[:size])
    # print (numpy.argmax(rslt), numpy.max(rslt))

    # rslt_real = util.cpaAtSin(dataTest[1].eval(), listTest)
    # print (numpy.argmax(rslt_real), numpy.max(rslt_real))

