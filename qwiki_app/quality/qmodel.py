# coding: utf-8
"""
implement static assessment classification model using theano
"""
import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from collections import OrderedDict
from django.conf import settings

rank_map = {"fa": 6, "ga": 5, "b": 4, "c": 3, "start": 2, "stub": 1}
rev_rank_map = {5: "fa", 4: "ga", 3: "b", 2: "c", 1: "start", 0: "stub"}

SEED = 123
numpy.random.seed(SEED)


def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)


def _p(pp, name):
    return '%s_%s' % (pp, name)


def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def param_init_lstm(options, params, prefix='lstm'):
    """
    Init the LSTM parameter:
    """
    W = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params


def init_params(options):
    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()

    # input to lstm input
    params['I2l'] = 0.01 * numpy.random.randn(options['dim_input'],
                                              options['dim_proj']).astype(config.floatX)
    params['lb'] = numpy.zeros((options['dim_proj'],)).astype(config.floatX)

    # lstm
    params = param_init_lstm(options, params)

    # hidden
    params['Uh'] = 0.01 * numpy.random.randn(options['dim_proj'] + options['dim_text'],
                                             options['dim_proj'] + options['dim_text']).astype(config.floatX)
    params['bh'] = numpy.zeros((options['dim_proj'] + options['dim_text'],)).astype(config.floatX)

    # classifier
    params['U'] = 0.01 * numpy.random.randn(options['dim_proj'] + options['dim_text'],
                                            options['ydim']).astype(config.floatX)
    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)

    return params


def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj


def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = options['dim_proj']
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0][-1], tensor.tanh(rval[1])


def build_model(tparams, options):
    trng = RandomStreams(SEED)
    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x = tensor.tensor3('x', dtype=config.floatX)
    x_text = tensor.matrix('x_text', dtype=config.floatX)
    mask = tensor.matrix('mask', dtype=config.floatX)
    y = tensor.vector('y', dtype='int64')
    n_samples = x.shape[1]

    # input to lstm input
    lx = tensor.dot(x, tparams['I2l']) + tparams['lb']

    # lstm
    proj, tanhcs = lstm_layer(tparams, lx, options, prefix='lstm', mask=mask)

    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)

    concatenated = tensor.concatenate([proj, x_text], axis=1)
    hidden = tensor.tanh(tensor.dot(concatenated, tparams['Uh']) + tparams['bh'])
    pred = tensor.nnet.softmax(tensor.dot(hidden, tparams['U']) + tparams['b'])

    f_pred_prob = theano.function([x, mask, x_text], pred, name='f_pred_prob')
    f_pred = theano.function([x, mask, x_text], pred.argmax(axis=1), name='f_pred')

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    cost = -tensor.log(pred[tensor.arange(n_samples), y] + off).mean()
    # cost = -tensor.sum(tensor.log(pred[tensor.arange(n_samples), y] + off))

    return x, mask, y, x_text, f_pred_prob, f_pred, cost


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def get_quality(
        input,
        reload_model=settings.QMODEL_DIR + "/model_combine_6_64_50.npz",  # Path to a saved model we want to start from.
        dim_input=6,
        dim_proj=64,
        dim_text=11,
        ydim=6,
        maxlen=50,
        noise_std=0.,
        use_dropout=True,
):
    # Model options
    model_options = locals().copy()

    params = init_params(model_options)
    load_params(reload_model, params)
    tparams = init_tparams(params)

    (x, mask, y, x_text, f_pred_prob, f_pred, cost) = build_model(tparams, model_options)

    y_pred = f_pred(input[0], input[1], input[2])

    return rev_rank_map[y_pred[0]].upper()


if __name__ == '__main__':
    from feature import get_data
    input = get_data("Wikipedia")

    y_pred = get_quality(input, reload_model="../static/model_combine_6_64_50.npz")
    print y_pred
