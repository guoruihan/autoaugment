# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.Graph().as_default()
session = tf.compat.v1.Session(graph=tf.get_default_graph(),config=config)

from tensorflow.keras import models, layers, datasets, utils, backend, optimizers, initializers
backend.set_session(session)

from keras_tqdm import TQDMCallback
import numpy as np

best_subpolicies = []
attack_subpolicy = None
(Xtr, ytr), (Xts, yts) = datasets.cifar10.load_data()
ytr = utils.to_categorical(ytr)
yts = utils.to_categorical(yts)
CHILD_BATCH_SIZE = 4096
CHILD_EPOCHS = 200

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
    def evaluate(self, X, y):
        return self.model.evaluate(X, y, verbose=0)[1]

if __name__ == "__main__":
    raw = TargetModel(Xtr.shape[1:])
    test = TargetModel(Xtr.shape[1:])
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