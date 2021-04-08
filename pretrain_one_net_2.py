import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os.path as osp
import logging
import sys
from utils import OfficeHomeImage
from model import ResNetFc
from model import get_cls_loss

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

MAIN_DIR = os.path.dirname(os.getcwd())

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default=osp.join(MAIN_DIR, "dataset/office"))
parser.add_argument("-s1", default="amazon")
parser.add_argument("-t", default="dslr")
parser.add_argument("--batch_size", default=32)
parser.add_argument("--shuffle", default=True)
parser.add_argument("--num_workers", default=8)
parser.add_argument("--snapshot", default=osp.join(MAIN_DIR, "MSDA/snapshot"))
parser.add_argument("--lr", default=0.0001)
parser.add_argument("--beta1", default=0.9)
parser.add_argument("--beta2", default=0.999)
parser.add_argument("--num_classes", default=31)
parser.add_argument("--threshold", default=0.9)
parser.add_argument("--log_interval", default=5)
parser.add_argument("--cls_epoches", default=100)
parser.add_argument("--gan_epoches", default=5)
args = parser.parse_args()

data_root = args.data_root
batch_size = args.batch_size
shuffle = args.shuffle
num_workers = args.num_workers
snapshot = args.snapshot
lr = args.lr
beta1 = args.beta1
beta2 = args.beta2
num_classes = args.num_classes
threshold = args.threshold
log_interval = args.log_interval
cls_epoches = args.cls_epoches
gan_epoches = args.gan_epoches


def get_log(file_name):
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fh = logging.FileHandler(file_name, mode='a')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


log_dir = './log/'
log_file = osp.join(log_dir, os.path.abspath(__file__).split('/')[-1].split('.')[0] + '.txt')
if os.path.isfile(log_file):
    os.remove(log_file)
logger = get_log(log_file)

def print_log(epoch, epoches, lr, l1, l2, l3, l4, l5, l6, l7, l8, flag):
    logger.info("Epoch [%d/%d] lr: %f, s1_cls_loss: %.4f, s2_cls_loss: %.4f, s1_t_dis_loss: %.4f, " \
                "s2_t_dis_loss: %.4f, s1_t_confusion_loss_s1: %.4f, s1_t_confusion_loss_t: %.4f, " \
                "s2_t_confusion_loss_s2: %.4f, s2_t_confusion_loss_t: %.4f, selected_source: %s" \
                % (epoch, epoches, lr, l1, l2, l3, l4, l5, l6, l7, l8, flag))



s1_root = os.path.join(data_root, args.s1, "images")
s1_label = os.path.join(data_root, args.s1, "label.txt")
t_root = os.path.join(data_root, args.t, "images")
t_label = os.path.join(data_root, args.t, "label.txt")
s1_set = OfficeHomeImage(s1_root, s1_label, split="train")
t_set_test = OfficeHomeImage(t_root, t_label, split="test")

assert len(s1_set) == 2817
assert len(t_set_test) == 498

s1_loader_raw = torch.utils.data.DataLoader(s1_set, batch_size=batch_size,
                                            shuffle=shuffle, num_workers=num_workers, drop_last=True)
t_loader_test = torch.utils.data.DataLoader(t_set_test, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers)


base_net=ResNetFc(resnet_name="ResNet50", use_bottleneck=False, bottleneck_dim=256, new_cls=True, class_num=31)
#base_net = nn.DataParallel(base_net)
base_net=base_net.cuda()

def inv_lr_scheduler(optimizer, iter_num, gamma, power, lr=0.001, weight_decay=0.0005):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = lr * (1 + gamma * iter_num) ** (-power)
    i=0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = weight_decay * param_group['decay_mult']
        i+=1

    return optimizer


parameter_list = base_net.get_parameters()
optim_base_net = optim.SGD(parameter_list, lr=0.001, momentum=0.9,weight_decay=0.0005, nesterov=True)
param_lr=[]
for param_group in optim_base_net.param_groups:
    param_lr.append(param_group["lr"])

count = 0
max_correct = 0
max_epoch = 0

iter_num=0
for cls_epoch in range(cls_epoches):  # cls_epoches
    base_net.train()
    s1_loader = iter(s1_loader_raw)

    for i, (s1_imgs, s1_labels) in enumerate(s1_loader):
        s1_imgs, s1_labels = Variable(s1_imgs.cuda()), Variable(s1_labels.cuda())
        optim_base_net = inv_lr_scheduler(optim_base_net, iter_num, lr=0.0003, gamma=0.001, power=0.75)
        iter_num=iter_num+1
        optim_base_net.zero_grad()
        _, s1_cls, _ = base_net(s1_imgs)
        s1_t_cls_loss = nn.CrossEntropyLoss()(s1_cls, s1_labels)
        s1_t_cls_loss.backward()
        optim_base_net.step()
        if (iter_num + 1) % 10 == 0:
            base_net.eval()

            correct = 0
            for (imgs, labels) in t_loader_test:
                imgs = Variable(imgs.cuda())
                _, s1_cls, _ = base_net(imgs)
                s1_cls = F.softmax(s1_cls, dim=1)
                s1_cls = s1_cls.data.cpu().numpy()
                res = s1_cls
                pred = np.argmax(res, axis=1)
                labels = labels.numpy()
                correct += np.equal(labels, pred).sum()
            current_accuracy = correct * 1.0 / len(t_set_test)
            current_accuracy = current_accuracy
            logger.info("Current accuracy is:{}".format(current_accuracy))

            torch.save(base_net.state_dict(), os.path.join(snapshot, "base_net" + ".pth"))

            if current_accuracy >= max_correct:
                max_correct = current_accuracy
                max_epoch = cls_epoch


logger.info("max_correct is :{}".format(str(max_correct)))
logger.info("max_epoch is :{}".format(str(max_epoch + 1)))
