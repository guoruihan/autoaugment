import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.Graph().as_default()
session = tf.compat.v1.Session(graph=tf.get_default_graph(),config=config)

from tensorflow.keras import models, layers, datasets, utils, backend, optimizers, initializers
backend.set_session(session)

from tqdm.keras import TqdmCallback
from tqdm import tqdm
import numpy as np

from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod, MomentumIterativeMethod, DeepFool, LBFGS

(Xtr, ytr), (Xts, yts) = datasets.cifar10.load_data()
Xtr = Xtr.astype(np.float32)
Xts = Xts.astype(np.float32)
ytr = utils.to_categorical(ytr)
yts = utils.to_categorical(yts)
CHILD_BATCH_SIZE = 64
CHILD_EPOCHS = 120

class TargetModel:
    def __init__(self, input_shape):
        self.model = self.create_model(input_shape)
        optimizer = optimizers.SGD(decay=1e-3)
        self.model.compile(optimizer, 'categorical_crossentropy', ['accuracy'])
    def create_model(self, input_shape):
        x = input_layer = layers.Input(shape=input_shape)
        x = layers.Conv2D(32, 3, activation='relu')(x)
        x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(10, activation='softmax')(x)
        return models.Model(input_layer, x)
    def fit(self, X, y, subpolicies=None):
        if subpolicies is not None:
            which = np.random.randint(len(subpolicies), size=len(X))
            for i, subpolicy in enumerate(subpolicies):
                X[which == i] = subpolicy(X[which == i])
        X = X.astype(np.float32) / 255
        callback = TqdmCallback(leave=False, file=sys.stdout, verbose=0)
        callback.on_train_batch_begin = callback.on_batch_begin
        callback.on_train_batch_end = callback.on_batch_end
        self.model.fit(X, y, CHILD_BATCH_SIZE, CHILD_EPOCHS, verbose=0, callbacks=[callback])
        return self
    def evaluate(self, X, y):
        return self.model.evaluate(X, y, verbose=0)[1]

if __name__ == "__main__":
    raw = TargetModel(Xtr.shape[1:])
    test = TargetModel(Xtr.shape[1:])

    raw_wrap = KerasModelWrapper(raw.model)
    raw_fgsm_ = FastGradientMethod(raw_wrap, sess=session)
    def fgsm(X, which, prob, magn):
        wrapped = FastGradientMethod(KerasModelWrapper(which.model), sess=session)
        X = X.copy()
        idx = np.random.uniform(size=len(X))
        idx = np.where(idx < prob)[0]
        for i in tqdm(range(0, len(idx), CHILD_BATCH_SIZE), desc=f'batch: ', leave=False):
            tensor = tf.convert_to_tensor(X[idx[i:i + CHILD_BATCH_SIZE]])
            init = tf.global_variables_initializer()
            session.run(init)
            tensor = wrapped.generate(tensor, eps=0.1 * magn)
            X[idx[i:i + CHILD_BATCH_SIZE]] = session.run(tensor)
        return X
    def df(X, which, prob, magn):
        wrapped = DeepFool(KerasModelWrapper(which.model), sess=session)
        X = X.copy()
        idx = np.random.uniform(size=len(X))
        idx = np.where(idx < prob)[0]
        for i in tqdm(range(0, len(idx), CHILD_BATCH_SIZE), desc=f'batch: ', leave=False):
            tensor = tf.convert_to_tensor(X[idx[i:i + CHILD_BATCH_SIZE]])
            init = tf.global_variables_initializer()
            session.run(init)
            tensor = wrapped.generate(tensor, clip_min=0., clip_max=magn * 0.3 + 0.3)
            X[idx[i:i + CHILD_BATCH_SIZE]] = session.run(tensor)
        return X
    def mim(X, which, prob, magn):
        wrapped = MomentumIterativeMethod(KerasModelWrapper(which.model), sess=session)
        X = X.copy()
        idx = np.random.uniform(size=len(X))
        idx = np.where(idx < prob)[0]
        for i in tqdm(range(0, len(idx), CHILD_BATCH_SIZE), desc=f'batch: ', leave=False):
            tensor = tf.convert_to_tensor(X[idx[i:i + CHILD_BATCH_SIZE]])
            init = tf.global_variables_initializer()
            session.run(init)
            tensor = wrapped.generate(tensor, eps=0.1 * magn)
            X[idx[i:i + CHILD_BATCH_SIZE]] = session.run(tensor)
        return X
    def lbfgs(X, which):
        wrapped = LBFGS(KerasModelWrapper(which.model), sess=session)
        X = X.copy()
        for i in tqdm(range(0, len(X), CHILD_BATCH_SIZE), desc=f'batch: ', leave=False):
            tensor = tf.convert_to_tensor(X[i:i + CHILD_BATCH_SIZE])
            tensor = wrapped.generate(tensor, eps=0.1)
            X[i:i + CHILD_BATCH_SIZE] = session.run(tensor)
        return X

    best_subpolicies = [
        lambda X: df(df(X, test, 0.1, 0.6), test, 0.1, 0.7),
        lambda X: mim(fgsm(X, test, 1.0, 0.1), test, 0.7, 0.4),
        lambda X: df(fgsm(X, test, 0.1, 0.2), test, 0.9, 0.9),
    ]
    attack_raw = lambda X: lbfgs(X, raw)
    attack_test = lambda X: lbfgs(X, test)
    print('\033[34mTrain raw model:\033[0m')
    # raw.fit(Xtr, ytr)
    print('\033[34mTrain test model:\033[0m')
    test.fit(Xtr, ytr, subpolicies=best_subpolicies)
    print('\033[34mAttack test dataset:\033[0m')
    Xraw = attack_raw(Xts)
    Xtest = attack_test(Xts)
    print('\033[34mRaw  model on   raw    testcase:', end='')
    raw_raw = raw.evaluate(Xts, yts)
    print(f'\033[32m {raw_raw}\033[0m')
    print('\033[34mRaw  model on attacked testcase:', end='')
    raw_attacked = raw.evaluate(Xraw, yts)
    print(f'\033[32m {raw_attacked}\033[0m')
    print('\033[34mTest model on   raw    testcase:', end='')
    test_raw = raw.evaluate(Xts, yts)
    print(f'\033[32m {test_raw}\033[0m')
    print('\033[34mTest model on attacked testcase:', end='')
    test_attacked = raw.evaluate(Xtest, yts)
    print(f'\033[32m {test_attacked}\033[0m')
    print(f'''
Test result:
Raw  model: {raw_raw:.4f} -> {raw_attacked:.4f}, reduced {raw_attacked - raw_raw:.4f}
Test model: {test_raw:.4f} -> {test_attacked:.4f}, reduced {test_attacked - test_raw:.4f}''')