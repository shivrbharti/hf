import sys, random
import numpy as np
import theano
import theano.tensor as T
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import cPickle as pickle
from collections import OrderedDict
from sklearn.metrics import roc_auc_score

def extract_params(zipped):
    new_params = OrderedDict()
    for key, value in zipped.items():
        new_params[key] = value.get_value()
    return new_params

def floatX(data):
    return np.asarray(data, dtype=config.floatX)

def random_weight(dim1, dim2, left=-0.1, right=0.1):
    return np.random.uniform(left, right, (dim1, dim2)).astype(config.floatX)

def initialize_params(options):
    params = OrderedDict()

    input_dim = options['inputDimSize']
    hidden_dim = options['hiddenDimSize']

    params['W_emb'] = np.eye(input_dim, dtype=config.floatX)

    params['W_gru'] = random_weight(input_dim+1, 3*hidden_dim)
    params['U_gru'] = random_weight(hidden_dim, 3*hidden_dim)
    params['b_gru'] = np.zeros(3*hidden_dim).astype(config.floatX)

    params['W_logistic'] = random_weight(hidden_dim, 1)
    params['b_logistic'] = np.zeros((1,), dtype=config.floatX)

    return params

def initialize_tparams(params):
    tparams = OrderedDict()
    for key, value in params.items():
        if key == 'W_emb': continue
        tparams[key] = theano.shared(value, name=key)
    return tparams

def dropout(state_before, use_noise, trng):
    proj = T.switch(use_noise, (state_before * trng.binomial(state_before.shape, p=0.5, n=1, dtype=state_before.dtype)), state_before * 0.5)
    return proj

def slice(_x, n, dim):
    if _x.ndim == 3:
        return _x[:, :, n*dim:(n+1)*dim]
    return _x[:, n*dim:(n+1)*dim]

def gru(tparams, emb, options, mask=None):
    hidden_dim = options['hiddenDimSize']
    timesteps = emb.shape[0]
    if emb.ndim == 3: n_samples = emb.shape[1]
    else: n_samples = 1

    def stepFn(stepMask, wx, h, U_gru):
        uh = T.dot(h, U_gru)
        r = T.nnet.sigmoid(slice(wx, 0, hidden_dim) + slice(uh, 0, hidden_dim))
        z = T.nnet.sigmoid(slice(wx, 1, hidden_dim) + slice(uh, 1, hidden_dim))
        h_tilde = T.tanh(slice(wx, 2, hidden_dim) + r * slice(uh, 2, hidden_dim))
        h_new = z * h + ((1. - z) * h_tilde)
        h_new = stepMask[:, None] * h_new + (1. - stepMask)[:, None] * h
        return h_new

    Wx = T.dot(emb, tparams['W_gru']) + tparams['b_gru']
    results, updates = theano.scan(fn=stepFn, sequences=[mask, Wx], outputs_info=T.alloc(floatX(0.0), n_samples, hidden_dim), non
