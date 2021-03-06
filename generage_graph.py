import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import tensorflow as tf
import matplotlib.pyplot as plt # plt 用于显示图片
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.Graph().as_default()
session = tf.compat.v1.Session(graph=tf.get_default_graph(),config=config)
# print(tf.get_default_graph())
# print(session.graph)
import cleverhans

from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# print(tf.test.is_built_with_cuda())

from tensorflow.keras import models, layers, datasets, utils, backend, optimizers, initializers
backend.set_session(session)
from transformations import get_transformations
import PIL.Image
import numpy as np
import time
from keras_tqdm import TQDMCallback
from tqdm import tqdm

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
        ix = np.random.choice(len(Xtr), 2048 * 8, False)
        Xtr = Xtr[ix]
        ytr = ytr[ix]
    Xtr = Xtr.astype(np.float32)
    Xts = Xts.astype(np.float32)
    ytr = utils.to_categorical(ytr)
    yts = utils.to_categorical(yts)
    return (Xtr, ytr), (Xts, yts)

(Xtr, ytr), (Xts, yts) = get_dataset('cifar10', True)
transformations = get_transformations()
# Experiment parameters

LSTM_UNITS = 100

SUBPOLICIES = 5
SUBPOLICY_OPS = 2

OP_TYPES = 3
OP_PROBS = 11
OP_MAGNITUDES = 10

CHILD_BATCH_SIZE = 64
CHILD_BATCHES = len(Xtr) // CHILD_BATCH_SIZE # '/' means normal divide, and '//' means integeral divide

CHILD_EPOCHS = 12
CONTROLLER_EPOCHS = 5 # 15000 or 20000
id_map = {
    0 : 'fgsm',
    # # 0 : 'lbfgs',
    # 1 : 'cwl2',
    2 : 'df' ,
    # 3 : 'enm' ,
    1 : 'mim'
}
model_map = {
        'fgsm' : fgsm,
        # 'lbfgs' : lbfgs,
        # 'cwl2' : cwl2,
        'df' : df,
        # 'enm' : enm,
        'mim' : mim
    }

class Operation:
    def __init__(self, types_softmax, probs_softmax, magnitudes_softmax, argmax=False):
        # Ekin Dogus says he sampled the softmaxes, and has not used argmax
        # We might still want to use argmax=True for the last predictions, to ensure
        # the best solutions are chosen and make it deterministic.
        if argmax:
            self.type = types_softmax.argmax()
            t = transformations[self.type]
            self.prob = probs_softmax.argmax() / (OP_PROBS-1)
            m = magnitudes_softmax.argmax() / (OP_MAGNITUDES-1)
            self.magnitude = m*(t[2]-t[1]) + t[1]
        else:
            self.type = np.random.choice(OP_TYPES, p=types_softmax)
            t = transformations[self.type]
            self.prob = np.random.choice(np.linspace(0, 1, OP_PROBS), p=probs_softmax)
            self.magnitude = np.random.choice(np.linspace(t[1], t[2], OP_MAGNITUDES), p=magnitudes_softmax)
            self.model = wrap
        self.transformation = t[0]

    def __call__(self, X):
        name = id_map[self.type]
        idx = np.random.uniform(size=len(X))
        idx = np.where(idx < self.prob)[0]
        print(len(X), self.prob, len(idx))
        for i in range(0, len(idx), CHILD_BATCH_SIZE):
            tensor = tf.convert_to_tensor(X[idx[i:i + CHILD_BATCH_SIZE]])
            tensor = self.transformation(tensor, self.magnitude, model_map[name])
            X[idx[i:i + CHILD_BATCH_SIZE]] = session.run(tensor)
        return X

    def __str__(self):
        return 'Operation %2d (P=%.3f, M=%.3f)' % (self.type, self.prob, self.magnitude)

class Subpolicy:
    def __init__(self, *operations):
        self.operations = operations

    def __call__(self, X):
        for op in self.operations:
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
        self.scale = tf.placeholder(tf.float32, ()) #不确定大小的占位符，最后会传入最终的值
        #print(self.model.outputs) #有一列model，输出结果也是一列
        #print(self.model.trainable_weights) #不是很懂
        self.grads = tf.gradients(self.model.outputs #模型输出张量的列表
                                  , self.model.trainable_weights)#可以训练的变量list
        #print(self.grads)
        # negative for gradient ascent
        self.grads = [g * (-self.scale) for g in self.grads]
        self.grads = zip(self.grads, self.model.trainable_weights) #直积
        self.optimizer = tf.train.GradientDescentOptimizer(0.00035).apply_gradients(self.grads)

    def create_model(self):
        # Implementation note: Keras requires an input. I create an input and then feed
        # zeros to the network. Ugly, but it's the same as disabling those weights.
        # Furthermore, Keras LSTM input=output, so we cannot produce more than SUBPOLICIES
        # outputs. This is not desirable, since the paper produces 25 subpolicies in the
        # end.
        input_layer = layers.Input(shape=(SUBPOLICIES, 1))
        init = initializers.RandomUniform(-0.1, 0.1)#生成均匀分布的随机数
        lstm_layer = layers.LSTM(
            LSTM_UNITS, #输出维度
            recurrent_initializer=init, #给偏置进行初始化操作的方法
            return_sequences=True,#返回全部输出的序列
            name='controller')(input_layer)
        outputs = []
        for i in range(SUBPOLICY_OPS):
            name = 'op%d-' % (i+1)
            outputs += [
                layers.Dense(OP_TYPES, activation='softmax', name=name + 't')(lstm_layer),
                layers.Dense(OP_PROBS, activation='softmax', name=name + 'p')(lstm_layer),
                layers.Dense(OP_MAGNITUDES, activation='softmax', name=name + 'm')(lstm_layer),
            ]
        #我们看到对每个操作里面建了三个网络，细节具体讨论
        return models.Model(input_layer, outputs)

    def fit(self, mem_softmaxes, mem_accuracies):
        #session = backend.get_session() #我们在session里面计算tensor
        min_acc = np.min(mem_accuracies)
        max_acc = np.max(mem_accuracies)
        dummy_input = np.zeros((1, SUBPOLICIES, 1))
        dict_input = {self.model.input: dummy_input}
        # FIXME: the paper does mini-batches (10)
        for softmaxes, acc in zip(mem_softmaxes, mem_accuracies): # learn this way to programming
            scale = (acc-min_acc) / (max_acc-min_acc)
            dict_outputs = {_output: s for _output, s in zip(self.model.outputs, softmaxes)}
            dict_scales = {self.scale: scale}
            #print(dict_scales)
            #print("rua")
            session.run(self.optimizer,#单个图元素
                        feed_dict={**dict_outputs, **dict_scales, **dict_input}) #将图元素映射到值的字典
        return self

    def predict(self, size):
        dummy_input = np.zeros((1, size, 1), np.float32)
        #没用的输入
        softmaxes = self.model.predict(dummy_input)
        # print("softmaxes")
        # print(softmaxes)
        # print("shape")
        # print(len(softmaxes))
        # convert softmaxes into subpolicies
        subpolicies = []
        for i in range(SUBPOLICIES):
            operations = []
            for j in range(SUBPOLICY_OPS):
                op = softmaxes[j*3:(j+1)*3]
                #print("op")
                #print(op)
                op = [o[0, i, :] for o in op]
                #print(op)
                operations.append(Operation(*op))
            subpolicies.append(Subpolicy(*operations))
        #print(subpolicies)
        return softmaxes, subpolicies

class Child:
    # architecture from: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
    def __init__(self, input_shape):
        self.model = self.create_model(input_shape)
        optimizer = optimizers.SGD(decay=1e-3)
        self.model.compile(optimizer, 'categorical_crossentropy', ['accuracy'])

    def create_model(self, input_shape):
        x = input_layer = layers.Input(shape=input_shape)
        x = layers.Conv2D(32, 3, activation='relu')(x)# take in a tensor and give out another tensor
        x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(10, activation='softmax')(x)
        return models.Model(input_layer, x)

    def fit(self, subpolicies, X, y):
        which = np.random.randint(len(subpolicies), size=len(X))
        for i, subpolicy in enumerate(subpolicies):
            X[which == i] = subpolicy(X[which == i])
        callback = TQDMCallback(leave_inner=False, leave_outer=False)
        callback.on_train_batch_begin = callback.on_batch_begin
        callback.on_train_batch_end = callback.on_batch_end
        self.model.fit(X, y, CHILD_BATCH_SIZE, CHILD_EPOCHS, verbose=0, callbacks=[callback])
        return self

    def evaluate(self, X, y):
        return self.model.evaluate(X, y, verbose=0)[1]

child = Child(Xtr.shape[1:])

wrap = KerasModelWrapper(child.model)
fgsm = FastGradientMethod(wrap, sess=session)
# lbfgs = LBFGS(wrap, sess=session)
# cwl2 = CarliniWagnerL2(wrap, sess=session)
df = DeepFool(wrap, sess=session)
# enm = ElasticNetMethod(wrap, sess=session)
mim = MomentumIterativeMethod(wrap, sess=session)

def attack(img):
    img = tf.expand_dims(img, 0)
    print("img",img)
    fgsm_params = {'eps': 0.3}
    return fgsm.generate(img, **fgsm_params)
tr = Xtr[0]
print(tr)
tr = np.array(tr,dtype='uint8')
print(tr)
plt.imshow(tr)  # 显示图片
plt.axis('off')  # 不显示坐标轴
plt.show()
print("rua2")

result = attack(tf.convert_to_tensor(tr.astype('float32')))
with session.as_default():
    result = result.eval()
    print(result)



plt.imshow(result)
plt.axis('off')
plt.show()

