import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from cleverhans.attacks import FastGradientMethod
from cleverhans.compat import flags
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

def FGSM(img, eps, fgsm):
    fgsm_params = {'eps': eps}
    return fgsm.generate(img, **fgsm_params)


def LBFGS(img, eps, lbfgs):
    lbfgs_params = {'eps': eps}
    return lbfgs.generate(img, **lbfgs_params)



def get_transformations():
    return [
        (FGSM, 0, 1.0),
        (LBFGS, 0, 1.0),
    ]

if __name__ == '__main__':
    tr = loadmat('../data/streetview/train_32x32.mat')
    imgs = np.moveaxis(tr['X'], -1, 0)
    transfs = get_transformations(imgs)
    for i in range(10):
        img2 = img = PIL.Image.fromarray(imgs[i])
        for t, min, max in transfs:
            v = np.random.rand()*(max-min) + min
            img2 = t(img2, v)
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        plt.imshow(img2)
        plt.show()
