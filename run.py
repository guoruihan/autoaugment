import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.Graph().as_default()
session = tf.compat.v1.Session(graph=tf.get_default_graph(),config=config)
import cleverhans

from tensorflow.python.client import device_lib

from tensorflow.keras import models, layers, datasets, utils, backend, optimizers, initializers
backend.set_session(session)
import transformations
import PIL.Image
import numpy as np
import time
from tqdm import tqdm
from tqdm.keras import TqdmCallback
import json

fgsm = None
lbfgs = None
cwl2 = None
df = None
enm = None
mim = None


from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import LBFGS
from cleverhans.attacks import MomentumIterativeMethod
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import DeepFool
from cleverhans.attacks import ElasticNetMethod
from cleverhans.dataset import MNIST
from cleverhans.loss import CrossEntropy
from cleverhans.train import train
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import cnn_model
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.utils_tf import model_eval

# datasets in the AutoAugment paper:
# CIFAR-10, CIFAR-100, SVHN, and ImageNet
# SVHN = http://ufldl.stanford.edu/housenumbers/

def get_dataset(dataset, reduced):
    if dataset == 'cifar10':
        (Xtr, ytr), (Xts, yts) = datasets.cifar10.load_data()
    elif dataset == 'cifar100':
        (Xtr, ytr), (Xts, yts) = datasets.cifar100.load_data()
    else:
        raise Exception('Unknown dataset %s' % dataset)
    if reduced:
        ix = np.random.choice(len(Xtr), 4000, False)
        Xtr = Xtr[ix]
        ytr = ytr[ix]
    Xtr = Xtr.astype(np.float32) / 255
    Xts = Xts.astype(np.float32) / 255
    ytr = utils.to_categorical(ytr)
    yts = utils.to_categorical(yts)
    return (Xtr, ytr), (Xts, yts)

(Xtr, ytr), (Xts, yts) = get_dataset('cifar10', True)
# Experiment parameters

LSTM_UNITS = 100

SUBPOLICIES = 3
SUBPOLICY_OPS = 2

OP_TYPES = 3
OP_PROBS = 11
OP_MAGNITUDES = 10

CHILD_BATCH_SIZE = 256

CHILD_EPOCHS = 200
CONTROLLER_EPOCHS = 1000 # 15000 or 20000
id_map = {
    0 : 'fgsm',
    # 1 : 'lbfgs',
    # 1 : 'cwl2',
    2 : 'df' ,
    # # 3 : 'enm' ,
    1 : 'mim'
}
wrapper_map = {
    'fgsm': transformations.FGSM,
    'df': transformations.DF,
    'mim': transformations.MIM,
}
model_map = {
    'fgsm' : fgsm,
    # 'lbfgs' : lbfgs,
    # 'cwl2' : cwl2,
    'df' : df,
    # 'enm' : enm,
    'mim' : mim
}
range_map = {
    'fgsm': [0, 0.1],
    'df': [0.3, 0.6],
    'mim': [0, 0.1],
}

class Operation:
    def __init__(self, types_softmax, probs_softmax, magnitudes_softmax, argmax=False):
        if argmax:
            self.type = types_softmax.argmax()
            self.prob = probs_softmax.argmax() / (OP_PROBS-1)
            self.magnitude = magnitudes_softmax.argmax() / (OP_MAGNITUDES-1)
        else:
            self.type = np.random.choice(OP_TYPES, p=types_softmax)
            self.prob = np.random.choice(np.linspace(0, 1, OP_PROBS), p=probs_softmax)
            self.magnitude = np.random.choice(np.linspace(0, 1, OP_MAGNITUDES), p=magnitudes_softmax)
        self.type = id_map[self.type]
        self.transformation = wrapper_map[self.type]

    def __call__(self, X):
        mi, ma = range_map[self.type]
        idx = np.random.uniform(size=len(X))
        idx = np.where(idx < self.prob)[0]
        for i in tqdm(range(0, len(idx), CHILD_BATCH_SIZE), desc='operation batch: ', file=sys.stdout, position=3, leave=False):
            tensor = tf.convert_to_tensor(X[idx[i:i + CHILD_BATCH_SIZE]])
            tensor = self.transformation(tensor, self.magnitude * (ma - mi) + mi, model_map[self.type])
            X[idx[i:i + CHILD_BATCH_SIZE]] = session.run(tensor)
        return X

    def __str__(self):
        return 'Operation %2d (P=%.3f, M=%.3f)' % (self.type, self.prob, self.magnitude)

class Subpolicy:
    def __init__(self, *operations):
        self.operations = operations

    def __call__(self, X):
        for op in tqdm(self.operations, desc='operation: ', file=sys.stdout, position=2, leave=False):
            X = op(X)
        return X

    def __str__(self):
        ret = ''
        for i, op in enumerate(self.operations):
            ret += str(op)
            if i < len(self.operations)-1:
                ret += '\n'
        return ret

class Controller:
    def __init__(self):
        self.model = self.create_model()
        self.scale = tf.placeholder(tf.float32, ())
        self.grads = tf.gradients(self.model.outputs, self.model.trainable_weights)
        self.grads = [g * (-self.scale) for g in self.grads]
        self.grads = zip(self.grads, self.model.trainable_weights)
        self.optimizer = tf.train.GradientDescentOptimizer(0.00035).apply_gradients(self.grads)

    def create_model(self):
        input_layer = layers.Input(shape=(SUBPOLICIES, 1))
        init = initializers.RandomUniform(-0.1, 0.1)
        lstm_layer = layers.LSTM(
            LSTM_UNITS,
            recurrent_initializer=init,
            return_sequences=True,
            name='controller')(input_layer)
        outputs = []
        for i in range(SUBPOLICY_OPS):
            name = 'op%d-' % (i+1)
            outputs += [
                layers.Dense(OP_TYPES, activation='softmax', name=name + 't')(lstm_layer),
                layers.Dense(OP_PROBS, activation='softmax', name=name + 'p')(lstm_layer),
                layers.Dense(OP_MAGNITUDES, activation='softmax', name=name + 'm')(lstm_layer),
            ]
        return models.Model(input_layer, outputs)

    def fit(self, mem_softmaxes, mem_accuracies):
        min_acc = np.min(mem_accuracies)
        max_acc = np.max(mem_accuracies)
        dummy_input = np.zeros((1, SUBPOLICIES, 1))
        dict_input = {self.model.input: dummy_input}
        for softmaxes, acc in zip(mem_softmaxes, mem_accuracies):
            scale = (acc-min_acc) / (max_acc-min_acc)
            dict_outputs = {_output: s for _output, s in zip(self.model.outputs, softmaxes)}
            dict_scales = {self.scale: scale}
            session.run(self.optimizer,
                        feed_dict={**dict_outputs, **dict_scales, **dict_input})
        return self

    def predict(self, size):
        dummy_input = np.zeros((1, size, 1), np.float32)
        softmaxes = self.model.predict(dummy_input)
        subpolicies = []
        for i in range(SUBPOLICIES):
            operations = []
            for j in range(SUBPOLICY_OPS):
                op = softmaxes[j*3:(j+1)*3]
                op = [o[0, i, :] for o in op]
                operations.append(Operation(*op))
            subpolicies.append(Subpolicy(*operations))
        return softmaxes, subpolicies

class Child:
    # architecture from: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
    def __init__(self, input_shape):
        self.model = self.create_model(input_shape)
        optimizer = optimizers.SGD(decay=1e-4)
        self.model.compile(optimizer, 'categorical_crossentropy', ['accuracy'])

    def create_model(self, input_shape):
        x = input_layer = layers.Input(shape=input_shape)
        x = layers.Conv2D(32, 3, activation='relu')(x)
        x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(10, activation='softmax')(x)
        return models.Model(input_layer, x)

    def fit(self, subpolicies, X, y):
        which = np.random.randint(len(subpolicies), size=len(X))
        for i, subpolicy in enumerate(tqdm(subpolicies, desc='subpolicy: ', file=sys.stdout, position=1, leave=False)):
            X[which == i] = subpolicy(X[which == i])
        callback = TqdmCallback(leave=False, file=sys.stdout)
        callback.on_train_batch_begin = callback.on_batch_begin
        callback.on_train_batch_end = callback.on_batch_end
        self.model.fit(X, y, CHILD_BATCH_SIZE, CHILD_EPOCHS, verbose=0, callbacks=[callback])
        return self

    def evaluate(self, X, y):
        return self.model.evaluate(X, y, verbose=0)[1]


mem_softmaxes = []
mem_accuracies = []

controller = Controller()
with open("subpolicy_result", "w"):
    pass

controller_iter = tqdm(range(CONTROLLER_EPOCHS), desc='epoch: ', position=0, file=sys.stdout)
for epoch in controller_iter:
    child = Child(Xtr.shape[1:])

    wrap = KerasModelWrapper(child.model)
    fgsm = FastGradientMethod(wrap, sess=session)
    lbfgs = LBFGS(wrap, sess=session)
    # cwl2 = CarliniWagnerL2(wrap, sess=session)
    df = DeepFool(wrap, sess=session)
    # enm = ElasticNetMethod(wrap, sess=session)
    mim = MomentumIterativeMethod(wrap, sess=session)

    model_map = {
        'fgsm' : fgsm,
        # 'lbfgs' : lbfgs,
        # 'cwl2' : cwl2,
        'df' : df,
        # 'enm' : enm,
        'mim' : mim
    }

    softmaxes, subpolicies = controller.predict(SUBPOLICIES)
    mem_softmaxes.append(softmaxes)

    tic = time.time()
    child.fit(subpolicies, Xtr, ytr)
    toc = time.time()
    accuracy = child.evaluate(Xts, yts)
    controller_iter.set_description(f'acc: {accuracy:.3f} | epoch: ')

    with open("subpolicy_result", "a") as f:
        ret = {
            'acc': float(accuracy),
            'subpolicy': [
                [{
                    'type': op.type,
                    'prob': float(op.prob),
                    'magnitude': float(op.magnitude),
                } for op in subpolicy.operations] for subpolicy in subpolicies ]
        }
        f.write(json.dumps(ret) + '\n')

    mem_accuracies.append(accuracy)

    if len(mem_softmaxes) > 5:
        controller.fit(mem_softmaxes, mem_accuracies)

print()
print('Best policies found:')
print()
_, subpolicies = controller.predict(SUBPOLICIES)
for i, subpolicy in enumerate(subpolicies):
    print('# Subpolicy %d' % (i+1))
    print(subpolicy)