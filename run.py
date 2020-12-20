import tensorflow as tf
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
from cleverhans.compat import flags
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
    ytr = utils.to_categorical(ytr)
    yts = utils.to_categorical(yts)
    return (Xtr, ytr), (Xts, yts)

(Xtr, ytr), (Xts, yts) = get_dataset('cifar10', True)
transformations = get_transformations()
# Experiment parameters

LSTM_UNITS = 100

SUBPOLICIES = 5
SUBPOLICY_OPS = 2

OP_TYPES = 6
OP_PROBS = 11
OP_MAGNITUDES = 10

CHILD_BATCH_SIZE = 128
CHILD_BATCHES = len(Xtr) // CHILD_BATCH_SIZE # '/' means normal divide, and '//' means integeral divide

CHILD_EPOCHS = 12
CONTROLLER_EPOCHS = 5 # 15000 or 20000
model_map = {
        'fgsm' : fgsm,
        'lbfgs' : lbfgs,
        'cwl2' : cwl2,
        'df' : df,
        'enm' : enm,
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

            # if(t[3] == 'fgsm'):
            #     self.model = fgsm
            # elif(t[3] == 'lbfgs'):
            #     self.model = lbfgs
            # elif(t[3] == 'cwl2'):
            #     self.model = cwl2
            # elif(t[3] == 'df'):
            #     self.model = df
            # elif(t[3] == 'enm'):
            #     self.model = enm
            # elif(t[3] == 'mim'):
            #     self.model = mim
            # else:
            #     assert(0)
            self.model = wrap
            #self.model = model_map.get(t[3])
        self.transformation = t[0]

    def __call__(self, X):
        _X = []
        x_use = None
        id = []
        tag = 0
        #print(X[0][0][0][0].type)
        X = X.astype(np.uint8)
        #print(X.shape)
        #assert(0)
        for x in X:
            if np.random.rand() < self.prob:
                #with session.graph.as_default():
                # print("tagx")
                # print(x)
                x = PIL.Image.fromarray(x)
                x = tf.image.resize_images(x, [32, 32])
                x.set_shape([32, 32, 3])
                x = tf.expand_dims(x,0)
                id.append(tag)
                if(x_use == None):
                    x_use = x
                else:
                    x_use = tf.concat([x_use,x],0)
            tag = tag + 1
        id.append(-1)
        x_use = tf.cast(x_use, tf.float32)
        fgsm = FastGradientMethod(self.model)
        fgsm_params = {'eps': self.magnitude}
        x_use = fgsm.generate(x_use, **fgsm_params)
            #assert(0)
        tag = 0
        npos = 0
        result = [tf.squeeze(tmp) for tmp in tf.split(x_use,[1 for i in range(x_use.shape[0])],0)]
        result = tuple(result)
        # print("rua1")
        # print(result)
        with session.as_default():
            result = session.run(result)
        result = list(result)
        # print("rua2")
        # print(result)
        for x in X:
            # print(tag)
            nv = x
            # print("x",x.shape)
            if(id[npos] == tag):
                #with session.as_default():
                nv=result[npos]
                #nv = tf.squeeze(nv)
                # print("x_use",x_use)
                # print("nv",nv)
                npos = npos + 1
            tag = tag + 1

            # with session.as_default():
            _X.append(np.array(nv))
        # print("tag1")
        # print(_X)
        # print("tag2")
        return np.array(_X)

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

# generator
def autoaugment(subpolicies, X, y):
    while True:
        ix = np.arange(len(X))
        np.random.shuffle(ix)
        for i in range(CHILD_BATCHES):
            _ix = ix[i*CHILD_BATCH_SIZE:(i+1)*CHILD_BATCH_SIZE]
            _X = X[_ix]
            _y = y[_ix]
            subpolicy = np.random.choice(subpolicies)
            _X = subpolicy(_X)
            _X = _X.astype(np.float32) / 255 # select from middle and put some subpolicy on that
            yield _X, _y

class Child:
    # architecture from: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
    def __init__(self, input_shape):
        self.model = self.create_model(input_shape)
        optimizer = optimizers.SGD(decay=1e-4)
        self.model.compile(optimizer, 'categorical_crossentropy', ['accuracy'])

    def create_model(self, input_shape):
        x = input_layer = layers.Input(shape=input_shape)
        x = layers.Conv2D(32, 3, activation='relu')(x)# take in a tensor and give out another tensor
        x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(10, activation='softmax')(x)
        return models.Model(input_layer, x)

    def fit(self, subpolicies, X, y):
        subpolicy = np.random.choice(subpolicies)
        # print(subpolicy)
#        assert(0)
        X = subpolicy(X)
        # print(X)
        X = X.astype(np.float32) / 255  # select from middle and put some subpolicy on that
        # print("tag")
        # assert(0)
        # print("base:",tf.get_default_graph())
        self.model.fit(X,y,CHILD_BATCH_SIZE, CHILD_EPOCHS, verbose=0, use_multiprocessing=False)
        return self

    def evaluate(self, X, y):
        return self.model.evaluate(X, y, verbose=0)[1]


mem_softmaxes = []
mem_accuracies = []

controller = Controller()



for epoch in range(CONTROLLER_EPOCHS):

    child = Child(Xtr.shape[1:])#(32,32,3)

    wrap = KerasModelWrapper(child.model)
    # fgsm = FastGradientMethod(wrap, sess=session)
    # lbfgs = LBFGS(wrap, sess=session)
    # cwl2 = CarliniWagnerL2(wrap, sess=session)
    # df = DeepFool(wrap, sess=session)
    # enm = ElasticNetMethod(wrap, sess=session)
    # mim = MomentumIterativeMethod(wrap, sess=session)

    model_map = {
        'fgsm': fgsm,
        'lbfgs': lbfgs,
        'cwl2': cwl2,
        'df': df,
        'enm': enm,
        'mim': mim
    }

    print('Controller: Epoch %d / %d' % (epoch+1, CONTROLLER_EPOCHS))
    softmaxes, subpolicies = controller.predict(SUBPOLICIES)
    for i, subpolicy in enumerate(subpolicies):
        print('# Sub-policy %d' % (i+1))
        print(subpolicy)
    mem_softmaxes.append(softmaxes)



    tic = time.time()
    child.fit(subpolicies, Xtr, ytr)
    toc = time.time()
    accuracy = child.evaluate(Xts, yts)
    print('-> Child accuracy: %.3f (elaspsed time: %ds)' % (accuracy, (toc-tic)))
    mem_accuracies.append(accuracy)# accuracy which was put into use

    if len(mem_softmaxes) > 5:
        # ricardo: I let some epochs pass, so that the normalization is more robust
        controller.fit(mem_softmaxes, mem_accuracies)
    print()

print()
print('Best policies found:')
print()
_, subpolicies = controller.predict(SUBPOLICIES)
for i, subpolicy in enumerate(subpolicies):
    print('# Subpolicy %d' % (i+1))
    print(subpolicy)