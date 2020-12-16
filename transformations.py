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

def CWL2(img, eps, cwl2):
    cwl2_params = {'confidence': eps} # 在所有参数中感觉是这个比较接近强度
    return cwl2.generate(img, **cwl2_params)

def DF(img, eps, df):
    df_params = {'clip_min':0., 'clip_max':eps} #感觉没有找到强度就用eps界定上下界了
    return df.generate(img, **df_params)

def ENM(img, eps, enm):
    enm_params = {'confidence' : eps}
    return enm.generate(img, **enm_params)




def get_transformations():
    return [
        (FGSM, 0, 1.0),
        (LBFGS, 0, 1.0),
        (CWL2, 0, 1.0),
        (DF, 0, 1.0),
        (ENM, 0, 1.0),
        
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
