import cleverhans
import numpy as np
from tensorflow.python.ops.gen_math_ops import xlogy
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from tqdm import tqdm
from cleverhans.utils import set_log_level
import logging
set_log_level(logging.ERROR)
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf
from cleverhans.model import CallableModelWrapper
from cleverhans.attacks import CarliniWagnerL2, FastGradientMethod, LBFGS, DeepFool, MaxConfidence, MomentumIterativeMethod, ProjectedGradientDescent, SpatialTransformationMethod, Noise
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

REDUCE = None
BATCH_SIZE = 4096
CHANNELS = [16, 32, 64, 32]
HIDDEN = 128
CHILD_EPOCHS = 1000
ADVERSIAL_EVERY = 1
SUBPOLICY_COUNT = 3
OPERATION_COUNT = 2
TYPE_COUNT = 3
PROB_COUNT = 11
MAGN_COUNT = 10

trainraw = torchvision.datasets.CIFAR10(root='data', train=True, download=True)
train_x, train_y = np.array(trainraw.data, dtype=float).transpose((0, 3, 1, 2)), np.array(trainraw.targets)
train_x /= 255
valraw = torchvision.datasets.CIFAR10(root='data', train=False, download=True)
val_x, val_y = np.array(valraw.data, dtype=float).transpose((0, 3, 1, 2)), np.array(valraw.targets)
val_x /= 255

if REDUCE is not None:
    train_idx = np.arange(len(train_x))
    np.random.shuffle(train_idx)
    train_x = train_x[train_idx[:REDUCE]]
    train_y = train_y[train_idx[:REDUCE]]
    val_idx = np.arange(len(val_x))
    np.random.shuffle(val_idx)
    val_x = val_x[val_idx[:REDUCE]]
    val_y = val_y[val_idx[:REDUCE]]

trainset = torch.utils.data.TensorDataset(torch.tensor(train_x, dtype=torch.float), torch.tensor(train_y, dtype=torch.long))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
valset = torch.utils.data.TensorDataset(torch.tensor(val_x, dtype=torch.float), torch.tensor(val_y, dtype=torch.long))
valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

class TestCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, CHANNELS[0], kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(CHANNELS[0], CHANNELS[1], kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(CHANNELS[1])
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(CHANNELS[1], CHANNELS[2], kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(CHANNELS[2], CHANNELS[3], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(CHANNELS[3])
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(CHANNELS[3] * 8 * 8, HIDDEN)
        self.relu5 = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(HIDDEN, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn2(x)
        x = self.relu4(x)
        x = self.pool2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


criterion = nn.CrossEntropyLoss()

def train_child(t, p, m, load_dict=False):
    # model = nn.DataParallel(TestCNN().cuda(1), device_ids=[1, 2, 3])
    raw_model = TestCNN().cuda(0)
    model = TestCNN().cuda(0)
    tf_model = convert_pytorch_model_to_tf(model)
    cleverhans_model = CallableModelWrapper(tf_model, output_layer='logits')
    session = tf.Session()
    x_op = tf.placeholder(tf.float32, shape=(None, 3, 32, 32))
    fgsm = FastGradientMethod(cleverhans_model, sess=session)
    # stm = SpatialTransformationMethod(cleverhans_model, sess=session)
    # cw2 = CarliniWagnerL2(cleverhans_model, sess=session)
    pgd = ProjectedGradientDescent(cleverhans_model, sess=session)
    noise = Noise(cleverhans_model, sess=session)
    mim = MomentumIterativeMethod(cleverhans_model, sess=session)
    df = DeepFool(cleverhans_model, sess=session)
    tf_raw_model = convert_pytorch_model_to_tf(raw_model)
    cleverhans_raw_model = CallableModelWrapper(tf_raw_model, output_layer='logits')
    pgd_raw = ProjectedGradientDescent(cleverhans_raw_model, sess=session)
    def fgsm_op(x, eps):
        att = fgsm.generate(x_op, eps=eps)
        return session.run(att, feed_dict={x_op: x})
    # def stm_op(x, eps):
    #     att = stm.generate(x_op, batch_size=len(x), dx_min=-0.1*eps, dx_max=0.1*eps, dy_min=-0.1*eps, dy_max=0.1*eps, angle_min=-30*eps, angle_max=30*eps)
    #     return session.run(att, feed_dict={x_op: x})
    # def cw2_op(x, eps):
    #     att = cw2.generate(x_op, max_iterations=3)
    def pgd_op(x, eps):
        att = pgd.generate(x_op, eps=eps, eps_iter=eps * 0.2, nb_iter=3)
        return session.run(att, feed_dict={x_op: x})
    def pgd_raw_op(x, eps):
        att = pgd_raw.generate(x_op, eps=eps, eps_iter=eps * 0.2, nb_iter=3)
        return session.run(att, feed_dict={x_op: x})
    def noise_op(x, eps):
        att = noise.generate(x_op, eps=eps)
        return session.run(att, feed_dict={x_op: x})
    def df_op(x):
        att = df.generate(x_op, nb_candidate=10, max_iter=3)
        return session.run(att, feed_dict={x_op: x})
    def mim_op(x, eps):
        att = mim.generate(x_op, eps=eps, eps_iter=eps * 0.2)
        return session.run(att, feed_dict={x_op: x})
    def attack_train(x):
        attacks = [fgsm_op, noise_op, mim_op]
        attacks_name = ['FGSM', 'Noise', 'MIM']
        eps = [[0.03, 0.3], [0.03, 0.3], [0.03, 0.3]]
        train_x_adv = x.copy()
        adv_type = np.random.randint(SUBPOLICY_COUNT, size=len(train_x_adv))
        for i, (ti, pi, mi) in enumerate(tqdm(zip(t, p, m), total=len(t), desc='Subpolicy: ', leave=False)):
            adv_i = train_x_adv[adv_type == i]
            for j, (tj, pj, mj) in enumerate(tqdm(zip(ti, pi, mi), total=len(ti), desc='Operation: ', leave=False)):
                tj, pj, mj = (*tj, *pj, *mj)
                adv_j = adv_i[np.random.randn(len(adv_i)) < pj]
                for i in tqdm(range(0, len(adv_j), BATCH_SIZE), desc=attacks_name[tj] + ': ', leave=False):
                    adv_j[i:][:BATCH_SIZE] = attacks[tj](adv_j[i:][:BATCH_SIZE], (mj + 1) / MAGN_COUNT * (eps[tj][1] - eps[tj][0]) + eps[tj][0])
        return train_x_adv
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    raw_optimizer = optim.SGD(raw_model.parameters(), lr=1e-3)
    train_x_adv = attack_train(train_x)
    adv_trainset = torch.utils.data.TensorDataset(torch.tensor(train_x_adv, dtype=torch.float), torch.tensor(train_y, dtype=torch.long))
    adv_trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    if load_dict:
        model.load_state_dict(torch.load('eval_runs/model.pt'))
        optimizer.load_state_dict(torch.load('eval_runs/optimizer.pt'))
        raw_model.load_state_dict(torch.load('eval_runs/raw_model.pt'))
        raw_optimizer.load_state_dict(torch.load('eval_runs/raw_optimizer.pt'))
    model.train()
    batch_tqdm = tqdm(adv_trainloader, leave=False)
    for x, y in batch_tqdm:
        optimizer.zero_grad()
        output = model(x.cuda(0))
        loss = criterion(output, y.cuda(0))
        loss.backward()
        optimizer.step()
        acc = torch.sum(output.cpu().argmax(axis=1) == y) / y.size(0)
        batch_tqdm.set_description(f'adv {loss:.3f} {acc:.3f}')
    batch_tqdm = tqdm(trainloader, leave=False)
    raw_model.train()
    for x, y in batch_tqdm:
        raw_optimizer.zero_grad()
        output = raw_model(x.cuda(0))
        loss = criterion(output, y.cuda(0))
        loss.backward()
        raw_optimizer.step()
        acc = torch.sum(output.cpu().argmax(axis=1) == y) / y.size(0)
        batch_tqdm.set_description(f'raw {loss:.3f} {acc:.3f}')
    with torch.no_grad():
        model.eval()
        batch_tqdm = tqdm(valloader, leave=False)
        tot_acc = 0
        for x, y in batch_tqdm:
            output = model(x.cuda(0))
            acc = float(torch.sum(output.cpu().argmax(axis=1) == y))
            tot_acc += acc
        adv_raw_acc = tot_acc / len(val_x)
        val_x_adv = np.zeros_like(val_x)
        for i in tqdm(range(0, len(val_x_adv), BATCH_SIZE), desc='PGD: ', leave=False):
            val_x_adv[i:][:BATCH_SIZE] = pgd_op(val_x[i:][:BATCH_SIZE], 0.01)
        adv_valset = torch.utils.data.TensorDataset(torch.tensor(val_x_adv, dtype=torch.float), torch.tensor(val_y, dtype=torch.long))
        adv_valloader = torch.utils.data.DataLoader(adv_valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        batch_tqdm = tqdm(adv_valloader, leave=False)
        tot_acc = 0
        for x, y in batch_tqdm:
            output = model(x.cuda(0))
            acc = float(torch.sum(output.cpu().argmax(axis=1) == y))
            tot_acc += acc
        adv_adv_acc = tot_acc / len(val_x)
        raw_model.eval()
        batch_tqdm = tqdm(valloader, leave=False)
        tot_acc = 0
        for x, y in batch_tqdm:
            output = raw_model(x.cuda(0))
            acc = float(torch.sum(output.cpu().argmax(axis=1) == y))
            tot_acc += acc
        raw_raw_acc = tot_acc / len(val_x)
        val_x_adv = np.zeros_like(val_x)
        for i in tqdm(range(0, len(val_x_adv), BATCH_SIZE), desc='PGD: ', leave=False):
            val_x_adv[i:][:BATCH_SIZE] = pgd_raw_op(val_x[i:][:BATCH_SIZE], 0.01)
        adv_valset = torch.utils.data.TensorDataset(torch.tensor(val_x_adv, dtype=torch.float), torch.tensor(val_y, dtype=torch.long))
        adv_valloader = torch.utils.data.DataLoader(adv_valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        batch_tqdm = tqdm(adv_valloader, leave=False)
        tot_acc = 0
        for x, y in batch_tqdm:
            output = raw_model(x.cuda(0))
            acc = float(torch.sum(output.cpu().argmax(axis=1) == y))
            tot_acc += acc
        raw_adv_acc = tot_acc / len(val_x)
    with open('eval_runs/acc.csv', 'a') as f:
        f.write(f'{adv_raw_acc},{adv_adv_acc},{raw_raw_acc},{raw_adv_acc}\n')
    print(f'adv {adv_raw_acc:.3f} -> {adv_adv_acc:.3f} | raw {raw_raw_acc:.3f} -> {raw_adv_acc:.3f}')
    torch.save(model.state_dict(), 'eval_runs/model.pt')
    torch.save(optimizer.state_dict(), 'eval_runs/optimizer.pt')
    torch.save(raw_model.state_dict(), 'eval_runs/raw_model.pt')
    torch.save(raw_optimizer.state_dict(), 'eval_runs/raw_optimizer.pt')

load_dict = os.path.isfile('eval_runs/model.pt')
if not load_dict:
    with open('eval_runs/acc.csv', 'w') as f:
        pass
train_child([[[0], [2]], [[1], [0]], [[1], [2]]],[[[7], [0]], [[4], [8]], [[7], [4]]],[[[5], [2]], [[4], [2]], [[5], [9]]], load_dict=load_dict)