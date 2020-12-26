import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt # plt 用于显示图片
from tensorflow.keras import models, layers, datasets, utils, backend, optimizers, initializers
import tensorflow as tf

from cleverhans.attacks import FastGradientMethod
from cleverhans.dataset import MNIST
from cleverhans.loss import CrossEntropy
from cleverhans.train import train
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import cnn_model
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.utils_tf import model_eval

from scipy.io import loadmat


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)
    return f

def FGSM(img, eps, model):
    # fgsm = FastGradientMethod(model)
    # print(tf.get_default_graph())
    # print(model.graph)
    # assert(0)
    # print(fgsm.sess.graph)
    # assert(0)
    fgsm_params = {'eps': eps}
    return model.generate(img, **fgsm_params)

#
# def LBFGS(img, eps, model):
#     # lbfgs = FastGradientMethod(model)
#     # print(tf.get_default_graph())
#     # print(model.graph)
#     # assert(0)
#     # print(lbfgs.sess.graph)
#     # assert(0)
#     lbfgs_params = {}
#     return model.generate(img, **lbfgs_params)

def CWL2(img, eps, model):
    # cwl2 = FastGradientMethod(model)
    # print(tf.get_default_graph())
    # print(model.graph)
    # assert(0)
    # print(cwl2.sess.graph)
    # assert(0)
    cwl2_params = {'confidence': eps} # 在所有参数中感觉是这个比较接近强度
    return model.generate(img, **cwl2_params)

def DF(img, eps, model):
    # df = FastGradientMethod(model)
    # print(tf.get_default_graph())
    # print(model.graph)
    # assert(0)
    # print(df.sess.graph)
    # assert(0)
    df_params = {'clip_min':0., 'clip_max':eps} #感觉没有找到强度就用eps界定上下界了
    return model.generate(img, **df_params)

def ENM(img, eps, model):
    # enm = FastGradientMethod(model)
    # print(tf.get_default_graph())
    # print(model.graph)
    # assert(0)
    # print(enm.sess.graph)
    # assert(0)
    enm_params = {'confidence' : eps}
    return model.generate(img, **enm_params)

def MIM(img, eps, model):
    # mim = FastGradientMethod(model)
    # print(tf.get_default_graph())
    # print(model.graph)
    # assert(0)
    # print(mim.sess.graph)
    # assert(0)
    mim_params = {'eps': eps}
    return model.generate(img, **mim_params)




def get_transformations():
    return [
        # (LBFGS, 0, 1.0, 'lbfgs'),
        (FGSM, 0, 1.0, 'fgsm'),
        # (CWL2, 0, 1.0, 'cwl2'),
        # (ENM, 0, 1.0, 'enm'),
        (MIM, 0, 1.0, 'mim'),
        (DF, 0, 1.0, 'df'),
    ]


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
    ytr = utils.to_categorical(ytr)
    yts = utils.to_categorical(yts)
    return (Xtr, ytr), (Xts, yts)

(Xtr, ytr), (Xts, yts) = get_dataset('cifar10', True)

def attack(img):
    pass


if __name__ == '__main__':
    print(Xtr[0].shape)
    plt.imshow(Xtr[0])  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    plt.show()
    tr = Xtr[0]

    plt.imshow(attack(tr))
    plt.axis('off')
    plt.show()
