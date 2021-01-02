import os, sys
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

controllerGraph = tf.Graph()
controllerSession = tf.compat.v1.Session(graph = controllerGraph,config = config)
import cleverhans

from tensorflow.python.client import device_lib

from tensorflow.keras import models, layers, datasets, utils, backend, optimizers, initializers
import transformations
import PIL.Image
import numpy as np
import time
from tqdm import tqdm
from tqdm.keras import TqdmCallback
import json

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

CHILD_BATCH_SIZE = 128

CHILD_EPOCHS = 100
CONTROLLER_EPOCHS = 2000 # 15000 or 20000
attack_types = ['fgsm', 'mim', 'df']
attack_func_map = {}
range_map = {
    'fgsm': [0.01, 0.1],
    'df': [1, 10],
    'mim': [0.01, 0.1],
}

def fgsm(model):
    wrap = KerasModelWrapper(model)
    att = FastGradientMethod(wrap, sess=session)
    def attack(X, eps):
        for i in tqdm(range(0, len(X), CHILD_BATCH_SIZE), desc=f'FGSM: ', file=sys.stdout, leave=False):
            # print(X[i:i+CHILD_BATCH_SIZE].shape)
            tensor = tf.convert_to_tensor(X[i:i + CHILD_BATCH_SIZE])
            tensor = att.generate(tensor, eps=eps)
            X[i:i + CHILD_BATCH_SIZE] = session.run(tensor)
    return attack

def lbfgs(model):
    wrap = KerasModelWrapper(model)
    att = LBFGS(wrap, sess=session)
    def attack(X):
        for i in tqdm(range(0, len(X), CHILD_BATCH_SIZE), desc=f'LBFGS: ', file=sys.stdout, leave=False):
            # print(X[i:i+CHILD_BATCH_SIZE].shape)
            tensor = tf.convert_to_tensor(X[i:i + CHILD_BATCH_SIZE])
            tensor = att.generate(tensor, batch_size=len(X[i:i + CHILD_BATCH_SIZE]), max_iterations=4, binary_search_steps=3)
            X[i:i + CHILD_BATCH_SIZE] = session.run(tensor)
    return attack

def df(model):
    wrap = KerasModelWrapper(model)
    att = DeepFool(wrap, sess=session)
    def attack(X, eps):
        for i in tqdm(range(0, len(X), CHILD_BATCH_SIZE), desc=f'DF: ', file=sys.stdout, leave=False):
            # print(X[i:i+CHILD_BATCH_SIZE].shape)
            tensor = tf.convert_to_tensor(X[i:i + CHILD_BATCH_SIZE])
            tensor = att.generate(tensor, nb_candidate=int(eps + 0.5))
            X[i:i + CHILD_BATCH_SIZE] = session.run(tensor)
            # import matplotlib.pyplot as plt
            # plt.imshow(X[i])
            # plt.show()
    return attack

def mim(model):
    wrap = KerasModelWrapper(model)
    att = MomentumIterativeMethod(wrap, sess=session)
    def attack(X, eps):
        for i in tqdm(range(0, len(X), CHILD_BATCH_SIZE), desc=f'MIM: ', file=sys.stdout, leave=False):
            # print(X[i:i+CHILD_BATCH_SIZE].shape)
            tensor = tf.convert_to_tensor(X[i:i + CHILD_BATCH_SIZE])
            tensor = att.generate(tensor, eps=eps, eps_iter=eps * 0.2)
            X[i:i + CHILD_BATCH_SIZE] = session.run(tensor)
    return attack

class Operation:
    def __init__(self, types_softmax, probs_softmax, magnitudes_softmax, argmax=False):
        if argmax:
            self.type = types_softmax.argmax()
            self.prob = probs_softmax.argmax() / (OP_PROBS-1)
            self.magnitude = (magnitudes_softmax.argmax() + 1) / OP_MAGNITUDES
        else:
            self.type = np.random.choice(OP_TYPES, p=types_softmax)
            self.prob = np.random.choice(np.linspace(0, 1, OP_PROBS), p=probs_softmax)
            self.magnitude = np.random.choice(np.linspace(0, 1, OP_MAGNITUDES), p=magnitudes_softmax)
        self.type = attack_types[self.type]
        self.transformation = attack_func_map[self.type]
        mi, ma = range_map[self.type]
        self.magnitude = self.magnitude * (ma - mi) + mi

    def __call__(self, X):
        idx = np.random.uniform(size=len(X))
        idx = np.where(idx < self.prob)[0]
        self.transformation(X[idx], self.magnitude)

    def __str__(self):
        return 'Operation %2d (P=%.3f, M=%.3f)' % (self.type, self.prob, self.magnitude)

class Subpolicy:
    def __init__(self, *operations):
        self.operations = operations

    def __call__(self, X):
        for op in tqdm(self.operations, desc='Operation: ', file=sys.stdout, leave=False):
            op(X)

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

    # def fit(self, mem_softmaxes, mem_accuracies):
    #     # min_acc = np.min(mem_accuracies)
    #     # max_acc = np.max(mem_accuracies)
    #     dummy_input = np.zeros((1, SUBPOLICIES, 1))
    #     dict_input = {self.model.input: dummy_input}
    #     for softmaxes, acc in zip(mem_softmaxes, mem_accuracies):
    #         # scale = (acc-min_acc) / (max_acc-min_acc)
    #         scale = acc
    #         dict_outputs = {_output: s for _output, s in zip(self.model.outputs, softmaxes)}
    #         dict_scales = {self.scale: scale}
    #         session.run(self.optimizer,
    #                     feed_dict={**dict_outputs, **dict_input, **dict_scales})
    #     return self
    def fit(self, softmaxes, accuracy):
        controllerSession.run(self.optimizer, feed_dict={
            **{_output: s for _output, s in zip(self.model.outputs, softmaxes)},
            self.model.input: np.zeros((1, SUBPOLICIES, 1)),
            self.scale: accuracy - 0.1,
        })

    def predict(self, size, argmax=False):
        dummy_input = np.zeros((1, size, 1), np.float32)
        softmaxes = self.model.predict(dummy_input)
        subpolicies = []
        for i in range(SUBPOLICIES):
            operations = []
            for j in range(SUBPOLICY_OPS):
                op = softmaxes[j*3:(j+1)*3]
                op = [o[0, i, :] for o in op]
                operations.append(Operation(*op, argmax=argmax))
            subpolicies.append(Subpolicy(*operations))
        return softmaxes, subpolicies

class Child:
    # architecture from: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
    def __init__(self, input_shape):
        self.model = self.create_model(input_shape)
        optimizer = optimizers.SGD(decay=1e-3)
        self.model.compile(optimizer, 'categorical_crossentropy', ['accuracy'])

    def create_model(self, input_shape):
        x = input_layer = layers.Input(shape=input_shape)
        x = layers.Conv2D(32, 3, activation='relu')(x)
        x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(10, activation='softmax')(x)
        return models.Model(input_layer, x)

    def fit(self, subpolicies, X, y, log_file, save_file):
        clean_x = X
        clean_y = y
        epoch_tqdm = tqdm(range(CHILD_EPOCHS), desc='Epoch: ', file=sys.stdout, leave=False)
        losses, accs = [], []
        for epoch in epoch_tqdm:
            for i in tqdm(range(0, len(X), CHILD_BATCH_SIZE), desc=f'Batch: ', file=sys.stdout, leave=False):
                self.model.train_on_batch(X[i:][:CHILD_BATCH_SIZE], y[i:][:CHILD_BATCH_SIZE])
            # return
            if epoch % 10 == 0:
                loss, acc = self.model.evaluate(Xts, yts, verbose=False)
                losses.append(float(loss))
                accs.append(float(acc))
                epoch_tqdm.set_description(f'Loss {loss:.3f} {acc:.3f} | Epoch {epoch}')
                X = clean_x.copy()
                y = clean_y.copy()
                shuffle_idx = np.arange(len(X))
                np.random.shuffle(shuffle_idx)
                X = X[shuffle_idx]
                y = y[shuffle_idx]
                which = np.random.randint(len(subpolicies), size=len(X))
                for i, subpolicy in enumerate(tqdm(subpolicies, desc='Subpolicy: ', file=sys.stdout, leave=False)):
                    subpolicy(X[which == i])
                # if epoch >= 30:
                #     import matplotlib.pyplot as plt
                #     print(X[0])
                #     plt.imshow(X[0])
                #     plt.show()
        with open(log_file, 'w') as f:
            f.write(json.dumps({
                'loss': losses,
                'acc': accs,
            }))
        self.model.save(save_file)

    def evaluate(self, X, y):
        return self.model.evaluate(X, y, verbose=0)[1]


mem_softmaxes = []
mem_accuracies = []

with controllerGraph.as_default():
    controller = Controller()
with open("subpolicy_result", "w"):
    pass

controller_iter = tqdm(range(CONTROLLER_EPOCHS), desc='Controller Epoch: ', position=0, file=sys.stdout, leave=False)
for epoch in controller_iter:
    tf.Graph().as_default()
    session = tf.compat.v1.Session(graph=tf.get_default_graph(), config=config)
    backend.set_session(session)

    child = Child(Xtr.shape[1:])
    attack_func_map = {
        'fgsm' : fgsm(child.model),
        'df' : df(child.model),
        'mim' : mim(child.model),
    }
    with controllerGraph.as_default():
        softmaxes, subpolicies = controller.predict(SUBPOLICIES, argmax=epoch % 10 == 9)


    # mem_softmaxes.append(softmaxes)

    child.fit(subpolicies, Xtr, ytr, log_file=f'runs/{epoch}.json', save_file=f'runs/{epoch}.h5')
    raw_accuracy = child.evaluate(Xts, yts)
    Xts_att = Xts.copy()
    lbfgs(child.model)(Xts_att)
    accuracy = child.evaluate(Xts_att, yts)
    controller_iter.set_description(f'Acc: {accuracy:.3f} | Controller Epoch')

    with open("subpolicy_result", "a") as f:
        ret = {
            'raw_acc': float(raw_accuracy),
            'acc': float(accuracy),
            'subpolicy': [
                [{
                    'type': op.type,
                    'prob': float(op.prob),
                    'magnitude': float(op.magnitude),
                } for op in subpolicy.operations] for subpolicy in subpolicies ]
        }
        f.write(json.dumps(ret) + '\n')

    # mem_accuracies.append(accuracy)
    with controllerGraph.as_default():
        controller.fit(softmaxes, accuracy)

    # if len(mem_softmaxes) > 5:
        # controller.fit(mem_softmaxes, mem_accuracies)
        # mem_accuracies = []
        # mem_softmaxes = []

print()
print('Best policies found:')
print()
_, subpolicies = controller.predict(SUBPOLICIES)
for i, subpolicy in enumerate(subpolicies):
    print('# Subpolicy %d' % (i+1))
    print(subpolicy)