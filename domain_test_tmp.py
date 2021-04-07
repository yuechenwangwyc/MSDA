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
from utils import OfficeImage
from model import Extractor, Classifier, Discriminator
from model import get_cls_loss, get_dis_loss, get_confusion_loss

os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

MAIN_DIR=os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default=osp.join(MAIN_DIR,"dataset/office"))
parser.add_argument("-s1", default="amazon")
parser.add_argument("-t", default="dslr")
parser.add_argument("--batch_size", default=32)
parser.add_argument("--shuffle", default=True)
parser.add_argument("--num_workers", default=8)
parser.add_argument("--steps", default=40)
parser.add_argument("--snapshot", default=osp.join(MAIN_DIR,"MSDA/A_W_2_D_Open/bvlc_A_W_2_D/snapshot","office"))
parser.add_argument("--s1_weight", default=0.5)
parser.add_argument("--lr", default=0.0001)
parser.add_argument("--beta1", default=0.9)
parser.add_argument("--beta2", default=0.999)
parser.add_argument("--gpu_id", default=1)
parser.add_argument("--num_classes", default=65)
parser.add_argument("--threshold", default=0.9)
parser.add_argument("--log_interval", default=5)
parser.add_argument("--cls_epoches", default=50)
parser.add_argument("--gan_epoches", default=5)
args = parser.parse_args()


data_root = args.data_root
batch_size = args.batch_size
shuffle = args.shuffle
num_workers = args.num_workers
steps = args.steps
snapshot = args.snapshot
s1_weight = args.s1_weight
lr = args.lr
beta1 = args.beta1
beta2 = args.beta2
gpu_id = args.gpu_id
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
log_file=osp.join(log_dir,os.path.abspath(__file__).split('/')[-1].split('.')[0]+'.txt')
if os.path.isfile(log_file):
    os.remove(log_file)
logger = get_log(log_file)



t_root = os.path.join(data_root, args.t, "images")
t_label = os.path.join(data_root, args.t, "label.txt")
t_set_test = OfficeImage(t_root, t_label, split="test")


t_loader_test = torch.utils.data.DataLoader(t_set_test, batch_size=batch_size,
    shuffle=False, num_workers=num_workers)


extractor = Extractor()
s1_classifier = Classifier(num_classes=31)





saved_state_dict = torch.load(osp.join(MAIN_DIR,"MSDA/A_W_2_D_Open/bvlc_A_W_2_D/pretrain/bvlc_extractor.pth"))
new_params = extractor.state_dict().copy()
for name, param in new_params.items():

    if name in saved_state_dict and param.size() == saved_state_dict[name].size():
        new_params[name].copy_(saved_state_dict[name])
    else:
        print('copy {}'.format(name))
extractor.load_state_dict(new_params)

saved_state_dict1 = torch.load(osp.join(MAIN_DIR,"MSDA/A_W_2_D_Open/bvlc_A_W_2_D/pretrain/bvlc_s1_cls.pth"))
new_params1 = s1_classifier.state_dict().copy()
for name, param in new_params1.items():

    if name in saved_state_dict1 and param.size() == saved_state_dict1[name].size():
        new_params1[name].copy_(saved_state_dict1[name])
    else:
        print('copy {}'.format(name))
s1_classifier.load_state_dict(new_params1)


extractor = nn.DataParallel(extractor)
extractor=extractor.cuda()
s1_classifier = nn.DataParallel(s1_classifier)
s1_classifier=s1_classifier.cuda()


count = 0
max_correct = 0
max_step = 0
max_epoch = 0

extractor.eval()
s1_classifier.eval()

correct = 0
for (imgs, labels) in t_loader_test:
    imgs = Variable(imgs.cuda())
    imgs_feature = extractor(imgs)
    s1_cls = s1_classifier(imgs_feature)
    s1_cls = F.softmax(s1_cls,dim=1)
    s1_cls = s1_cls.data.cpu().numpy()
    res = s1_cls
    pred = np.argmax(res,axis=1)
    labels = labels.numpy()
    correct += np.equal(labels, pred).sum()
current_accuracy = correct * 1.0 / len(t_set_test)
current_accuracy=current_accuracy.item()
logger.info( "Current accuracy is:{}".format(current_accuracy))


