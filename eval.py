import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1,3'

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.Graph().as_default()
session = tf.compat.v1.Session(graph=tf.get_default_graph(),config=config)

from tensorflow.keras import models, layers, datasets, utils, backend, optimizers, initializers
backend.set_session(session)

from keras_tqdm import TQDMCallback
import numpy as np

from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod

(Xtr, ytr), (Xts, yts) = datasets.cifar10.load_data()
Xtr = Xtr.astype(np.float32)
Xts = Xts.astype(np.float32)
ytr = utils.to_categorical(ytr)
yts = utils.to_categorical(yts)
CHILD_BATCH_SIZE = 64
CHILD_EPOCHS = 1

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
        callback = TQDMCallback(leave_inner=False, leave_outer=False)
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
    raw_fgsm_p = 0.5
    raw_fgsm_eps = 0.2
    def raw_fgsm(X):
        X = X.copy()
        idx = np.random.uniform(size=len(X))
        tensor = tf.convert_to_tensor(X[idx < raw_fgsm_p])
        tensor = raw_fgsm_.generate(tf.convert_to_tensor(X[idx < raw_fgsm_p]), eps=raw_fgsm_eps)
        # print(tensor)
        # quit()
        X[idx < raw_fgsm_p] = session.run(tensor)
        return X

    best_subpolicies = [raw_fgsm]
    attack_raw = raw_fgsm
    attack_test = raw_fgsm
    print('\033[34mTrain raw model:\033[0m')
    raw.fit(Xtr, ytr)
    print('\033[34mTrain test model:\033[0m')
    test.fit(Xtr, ytr, subpolicies=best_subpolicies)
    print('\033[34mAttack test dataset:\033[0m')
    Xat = attack_subpolicy(Xts)
    print('\033[34mRaw  model on   raw    testcase:', end='')
    raw_raw = raw.evaluate(Xts, yts)
    print(f'\033[32m {raw_raw}\033[0m')
    print('\033[34mRaw  model on attacked testcase:', end='')
    raw_attacked = raw.evaluate(Xat, yts)
    print(f'\033[32m {raw_attacked}\033[0m')
    print('\033[34mTest model on   raw    testcase:', end='')
    test_raw = raw.evaluate(Xts, yts)
    print(f'\033[32m {test_raw}\033[0m')
    print('\033[34mTest model on attacked testcase:', end='')
    test_attacked = raw.evaluate(Xat, yts)
    print(f'\033[32m {test_attacked}\033[0m')
    print(f'''
Test result:
Raw  model: {raw_raw:.4f} -> {raw_attacked:.4f}, reduced {raw_attacked - raw_raw:.4f}
Test model: {test_raw:.4f} -> {test_attacked:.4f}, reduced {test_attacked - test_raw:.4f}''')