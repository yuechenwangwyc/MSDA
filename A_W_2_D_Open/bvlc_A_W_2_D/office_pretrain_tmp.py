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
from utils import OfficeImage, LinePlotter
from model import Extractor, Classifier, Discriminator
from model import get_cls_loss, get_dis_loss, get_confusion_loss

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

MAIN_DIR=os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default=osp.join(MAIN_DIR,"dataset/office/"))
parser.add_argument("-s1", default="amazon")
parser.add_argument("-t", default="webcam")
parser.add_argument("--batch_size", default=32)
parser.add_argument("--shuffle", default=True)
parser.add_argument("--num_workers", default=8)
parser.add_argument("--steps", default=8)
parser.add_argument("--snapshot", default=osp.join(MAIN_DIR,"MSDA/A_W_2_D_Open/bvlc_A_W_2_D/snapshot/"))
parser.add_argument("--s1_weight", default=0.5)
parser.add_argument("--s2_weight", default=0.5)
parser.add_argument("--lr", default=0.00001)
parser.add_argument("--beta1", default=0.9)
parser.add_argument("--beta2", default=0.999)
parser.add_argument("--gpu_id", default=1)
parser.add_argument("--num_classes", default=31)
parser.add_argument("--threshold", default=0.9)
parser.add_argument("--log_interval", default=5)
parser.add_argument("--cls_epoches", default=10)
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


s1_root = os.path.join(data_root, args.s1, "images")
s1_label = os.path.join(data_root, args.s1, "label.txt")

t_root = os.path.join(data_root, args.t, "images")
t_label = os.path.join(data_root, args.t, "label.txt")



s1_set = OfficeImage(s1_root, s1_label, split="train")


t_set_test = OfficeImage(t_root, t_label, split="test")
t_set_test_s1 = OfficeImage(s1_root, s1_label, split="test")
assert len(s1_set) == 2817

#assert len(t_set_test) == 2817
s1_loader_raw = torch.utils.data.DataLoader(s1_set, batch_size=batch_size,
    shuffle=shuffle, num_workers=num_workers)

t_loader_test = torch.utils.data.DataLoader(t_set_test, batch_size=batch_size,
    shuffle=False, num_workers=num_workers)
t_loader_test_s1 = torch.utils.data.DataLoader(t_set_test_s1, batch_size=batch_size,
    shuffle=False, num_workers=num_workers)


extractor = Extractor()
s1_classifier = Classifier(num_classes=num_classes)

# extractor.load_state_dict(torch.load(osp.join(MAIN_DIR,"MSDA/A_W_2_D_Open/
# bvlc_A_W_2_D/pretrain/office/bvlc_extractor.pth")))
# s1_classifier.load_state_dict(torch.load(osp.join(MAIN_DIR,"MSDA/A_W_2_D_Open
# /bvlc_A_W_2_D/pretrain/office/bvlc_s1_cls.pth")))

s1_t_discriminator = Discriminator()


extractor = nn.DataParallel(extractor)
extractor=extractor.cuda()

s1_classifier = nn.DataParallel(s1_classifier)

s1_classifier=s1_classifier.cuda()



s1_t_discriminator = nn.DataParallel(s1_t_discriminator)
s1_t_discriminator=s1_t_discriminator.cuda()



def print_log(step, epoch, epoches, lr, l1, l2, l3, l4, l5, l6, l7, l8, flag, ploter, count):
    logger.info("Step [%d/%d] Epoch [%d/%d] lr: %f, s1_cls_loss: %.4f, s2_cls_loss: %.4f, s1_t_dis_loss: %.4f, " \
          "s2_t_dis_loss: %.4f, s1_t_confusion_loss_s1: %.4f, s1_t_confusion_loss_t: %.4f, " \
          "s2_t_confusion_loss_s2: %.4f, s2_t_confusion_loss_t: %.4f, selected_source: %s" \
          % (step, steps, epoch, epoches, lr, l1, l2, l3, l4, l5, l6, l7, l8, flag))


count = 0
max_correct = 0
max_step = 0
max_epoch = 0

'''
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
    #pred = res.argmax(axis=1)
    pred = np.argmax(res,axis=1)
    labels = labels.numpy()
    correct += np.equal(labels, pred).sum()
current_accuracy = correct * 1.0 / len(t_set_test)
current_accuracy=current_accuracy.item()
logger.info( "Current accuracy is:{}".format(current_accuracy))

correct = 0
for (imgs, labels) in t_loader_test_s1:
    imgs = Variable(imgs.cuda())
    imgs_feature = extractor(imgs)
    s1_cls = s1_classifier(imgs_feature)

    s1_cls = F.softmax(s1_cls,dim=1)

    s1_cls = s1_cls.data.cpu().numpy()

    res = s1_cls
    #pred = res.argmax(axis=1)
    pred = np.argmax(res,axis=1)
    labels = labels.numpy()
    correct += np.equal(labels, pred).sum()
current_accuracy = correct * 1.0 / len(t_set_test_s1)
current_accuracy=current_accuracy.item()
logger.info( "Current accuracy s1 is:{}".format(current_accuracy))

'''


ploter = LinePlotter(env_name="bvlc_A_W_2_D")
for step in range(steps):
    # Part 1: assign psudo-labels to t-domain and update the label-dataset

    # Part 2: train F1t, F2t with pseudo labels
    logger.info( "#################### Part2 ####################")
    extractor.train()
    s1_classifier.train()


    optim_extract = optim.Adam(extractor.parameters(), lr=lr, betas=(beta1, beta2))
    optim_s1_cls = optim.Adam(s1_classifier.parameters(), lr=lr, betas=(beta1, beta2))


    for cls_epoch in range(cls_epoches):#cls_epoches
        s1_loader= iter(s1_loader_raw)
        for i, (s1_imgs, s1_labels) in enumerate(s1_loader):

            s1_imgs, s1_labels = Variable(s1_imgs.cuda()), Variable(s1_labels.cuda())


            optim_extract.zero_grad()
            optim_s1_cls.zero_grad()

            s1_t_feature = extractor(s1_imgs)
            s1_t_cls = s1_classifier(s1_t_feature)
            s1_t_cls_loss = get_cls_loss(s1_t_cls, s1_labels)

            torch.autograd.backward([s1_t_cls_loss])

            optim_s1_cls.step()
            optim_extract.step()

            if (i+1) % 50 == 0:
                print_log(step+1, cls_epoch+1, cls_epoches, lr, s1_t_cls_loss.data[0], \
                           s1_t_cls_loss.data[0], 0, 0, 0, 0, 0, 0, "...", ploter, count)
                count += 1
    
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
        #pred = res.argmax(axis=1)
        pred = np.argmax(res,axis=1)
        labels = labels.numpy()
        correct += np.equal(labels, pred).sum()
    current_accuracy = correct * 1.0 / len(t_set_test)
    current_accuracy=current_accuracy.item()
    logger.info( "Current accuracy is:{}".format(current_accuracy))
    correct = 0
    for (imgs, labels) in t_loader_test_s1:
        imgs = Variable(imgs.cuda())
        imgs_feature = extractor(imgs)
        s1_cls = s1_classifier(imgs_feature)

        s1_cls = F.softmax(s1_cls, dim=1)

        s1_cls = s1_cls.data.cpu().numpy()

        res = s1_cls
        # pred = res.argmax(axis=1)
        pred = np.argmax(res, axis=1)
        labels = labels.numpy()
        correct += np.equal(labels, pred).sum()
    current_accuracy = correct * 1.0 / len(t_set_test_s1)
    current_accuracy = current_accuracy.item()
    logger.info("Current accuracy is:{}".format(current_accuracy))

    if current_accuracy >= max_correct:
        max_correct = current_accuracy
        max_step = step
        max_epoch = cls_epoch


            


logger.info("max_correct is :{}".format(str(max_correct)))
logger.info("max_step is :{}".format(str(max_step+1)))
logger.info("max_epoch is :{}".format(str(max_epoch+1)))
