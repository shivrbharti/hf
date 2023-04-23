import sys
import random
import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
import pickle

config = theano.config
theano.config.optimizer = 'fast_compile'


def np_floatX(data):
    return np.asarray(data, dtype=config.floatX)


def init_weights(shape, name):
    return theano.shared(np_floatX(np.random.randn(*shape) * 0.01), name=name)


def init_params(options):
    params = OrderedDict()
    params['W_emb'] = np_floatX(np.random.randn(options['inputDimSize'], options['embDimSize']) * 0.01)
    params['W_gru'] = np.concatenate([init_weights((options['embDimSize'], options['hiddenDimSize']), 'W_gru_in'),
                                      init_weights((options['hiddenDimSize'], options['hiddenDimSize']), 'W_gru_hid')],
                                     axis=1)
    params['U_gru'] = np.concatenate([init_weights((options['hiddenDimSize'], options['hiddenDimSize']), 'U_gru_in'),
                                      init_weights((options['hiddenDimSize'], options['hiddenDimSize']), 'U_gru_hid')],
                                     axis=1)
    params['b_gru'] = np.zeros((2 * options['hiddenDimSize'],), dtype=config.floatX)
    params['W_logistic'] = init_weights((options['hiddenDimSize'], 1), 'W_logistic')
    params['b_logistic'] = np.zeros((1,), dtype=config.floatX)

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for key, value in params.items():
        tparams[key] = theano.shared(value, name=key)
    return tparams


def unzip(zipped_params):
    new_params = OrderedDict()
    for key, value in zipped_params.items():
        new_params[key] = value.get_value()
    return new_params


def gru_layer(tparams, state_below, options, mask=None):
    n_timesteps = state_below.shape[0]
    n_samples = state_below.shape[1]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_):
        preact = T.dot(h_, tparams['U_gru'])
        preact += x_

        r = T.nnet.sigmoid(_slice(preact, 0, options['hiddenDimSize']))
        u = T.nnet.sigmoid(_slice(preact, 1, options['hiddenDimSize']))

        preactx = T.dot(h_, tparams['U_gru'][:, 2 * options['hiddenDimSize']:])
        preactx = preactx * r
        preactx = preactx + _slice(x_, 1, options['hiddenDimSize'])

        h = T.tanh(preactx)
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    state_below = (T.dot(state_below, tparams['W_gru']) + tparams['b_gru'])

    if mask is None:
        mask = T.alloc(1., state_below.shape[0], 1)

    rval, updates = theano.scan(_step,
                                 sequences=[mask,                                state_below],
                                outputs_info=[T.alloc(numpy_floatX(0.), n_samples, options['hiddenDimSize'])],
                                name='gru_layer',
                                n_steps=n_timesteps)
    return rval[-1]


def dropout_layer(state_before, use_noise, trng):
    proj = T.switch(use_noise, (state_before * trng.binomial(state_before.shape, p=0.5, n=1,
                                                              dtype=state_before.dtype)), state_before * 0.5)
    return proj


def build_model(tparams, options, Wemb):
    trng = T.shared_randomstreams.RandomStreams(123)
    use_noise = theano.shared(numpy_floatX(0.))

    x = T.matrix('x', dtype='int32')
    mask = T.matrix('mask', dtype=config.floatX)
    y = T.vector('y', dtype='int32')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    emb = Wemb[x.flatten()].reshape([n_timesteps, n_samples, options['embDimSize']])

    proj = gru_layer(tparams, emb, options, mask=mask)
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)

    p_y_given_x = T.nnet.sigmoid(T.dot(proj, tparams['W_logistic']) + tparams['b_logistic'])
    L = -(y * T.flatten(T.log(p_y_given_x)) + (1 - y) * T.flatten(T.log(1 - p_y_given_x)))
    cost = T.mean(L)

    if options['L2_reg'] > 0.:
        cost += options['L2_reg'] * (tparams['W_logistic'] ** 2).sum()

    return use_noise, x, mask, y, p_y_given_x, cost


if __name__ == '__main__':
	dataFile = sys.argv[1]
	labelFile = sys.argv[2]
	embFile = sys.argv[3]
	outFile = sys.argv[4]

	inputDimSize = 15954
	embDimSize = 100
	hiddenDimSize = 100
	max_epochs = 100
	L2_reg = 0.001
	batchSize = 100
	use_dropout = True

	inputDimSize = 100 #The number of unique medical codes
	embDimSize = 100 #The size of the code embedding
	hiddenDimSize = 100 #The size of the hidden layer of the GRU
	max_epochs = 100 #Maximum epochs to train
	L2_reg = 0.001 #L2 regularization for the logistic weight
	batchSize = 10 #The size of the mini-batch
	use_dropout = True #Whether to use a dropout between the GRU and the logistic layer

	train_GRU_RNN(dataFile=dataFile, labelFile=labelFile, embFile=embFile, outFile=outFile, inputDimSize=inputDimSize, embDimSize=embDimSize, hiddenDimSize=hiddenDimSize, max_epochs=max_epochs, L2_reg=L2_reg, batchSize=batchSize, use_dropout=use_dropout)
