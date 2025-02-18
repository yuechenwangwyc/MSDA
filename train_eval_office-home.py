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

MAIN_DIR=os.path.dirname(os.getcwd())

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default=osp.join(MAIN_DIR,"dataset/office-home"))
parser.add_argument("-s1", default="Art")
parser.add_argument("-s2", default="Product")
parser.add_argument("-s3", default="Real_World")
parser.add_argument("-t", default="Clipart")
parser.add_argument("--batch_size", default=32)
parser.add_argument("--shuffle", default=True)
parser.add_argument("--num_workers", default=8)
parser.add_argument("--steps", default=8)
parser.add_argument("--snapshot", default=osp.join(MAIN_DIR,"MSDA/A_W_2_D_Open/bvlc_A_W_2_D/snapshot","office-home"))
parser.add_argument("--s1_weight", default=0.5)
parser.add_argument("--s2_weight", default=0.5)
parser.add_argument("--s3_weight", default=0.5)
parser.add_argument("--lr", default=0.00001)
parser.add_argument("--beta1", default=0.9)
parser.add_argument("--beta2", default=0.999)
parser.add_argument("--gpu_id", default=1)
parser.add_argument("--num_classes", default=65)
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
s2_weight = args.s2_weight
s3_weight = args.s3_weight
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


s1_root = os.path.join(data_root, args.s1)
s1_label = os.path.join(data_root, args.s1+".txt")
s2_root = os.path.join(data_root, args.s2)
s2_label = os.path.join(data_root, args.s2+".txt")
s3_root = os.path.join(data_root, args.s3)
s3_label = os.path.join(data_root, args.s3+".txt")
t_root = os.path.join(data_root, args.t)
t_label = os.path.join(data_root, args.t+".txt")
s1_set = OfficeImage(s1_root, s1_label, split="train")
s2_set = OfficeImage(s2_root, s2_label, split="train")
s3_set = OfficeImage(s3_root, s3_label, split="train")
t_set = OfficeImage(t_root, t_label, split="train")
t_set_test = OfficeImage(t_root, t_label, split="test")
assert len(s1_set) == 2427
assert len(s2_set) == 4439
assert len(s3_set) == 4357
assert len(t_set) == 4365
assert len(t_set_test) == 4365
s1_loader_raw = torch.utils.data.DataLoader(s1_set, batch_size=batch_size,
    shuffle=shuffle, num_workers=num_workers, drop_last=True)
s2_loader_raw = torch.utils.data.DataLoader(s2_set, batch_size=batch_size,
    shuffle=shuffle, num_workers=num_workers, drop_last=True)
s3_loader_raw = torch.utils.data.DataLoader(s3_set, batch_size=batch_size,
    shuffle=shuffle, num_workers=num_workers, drop_last=True)
t_loader_raw = torch.utils.data.DataLoader(t_set, batch_size=batch_size,
    shuffle=shuffle, num_workers=num_workers, drop_last=True)
t_loader_test = torch.utils.data.DataLoader(t_set_test, batch_size=batch_size,
    shuffle=False, num_workers=num_workers)


extractor = Extractor()
s1_classifier = Classifier(num_classes=num_classes)
s2_classifier = Classifier(num_classes=num_classes)
s3_classifier = Classifier(num_classes=num_classes)
s1_t_discriminator = Discriminator()
s2_t_discriminator = Discriminator()
s3_t_discriminator = Discriminator()

extractor.load_state_dict(torch.load(osp.join(MAIN_DIR,"MSDA/A_W_2_D_Open/bvlc_A_W_2_D/pretrain/bvlc_extractor.pth")))
extractor = nn.DataParallel(extractor)
extractor=extractor.cuda()

s1_classifier.load_state_dict(torch.load(osp.join(MAIN_DIR,"MSDA/A_W_2_D_Open/bvlc_A_W_2_D/pretrain/office-home/bvlc_s1_cls.pth")))
s2_classifier.load_state_dict(torch.load(osp.join(MAIN_DIR,"MSDA/A_W_2_D_Open/bvlc_A_W_2_D/pretrain/office-home/bvlc_s2_cls.pth")))
s3_classifier.load_state_dict(torch.load(osp.join(MAIN_DIR,"MSDA/A_W_2_D_Open/bvlc_A_W_2_D/pretrain/office-home/bvlc_s3_cls.pth")))
s1_classifier = nn.DataParallel(s1_classifier)
s2_classifier = nn.DataParallel(s2_classifier)
s3_classifier = nn.DataParallel(s3_classifier)
s1_classifier=s1_classifier.cuda()
s2_classifier=s2_classifier.cuda()
s3_classifier=s3_classifier.cuda()


s1_t_discriminator = nn.DataParallel(s1_t_discriminator)
s1_t_discriminator=s1_t_discriminator.cuda()
s2_t_discriminator = nn.DataParallel(s2_t_discriminator)
s2_t_discriminator=s2_t_discriminator.cuda()
s3_t_discriminator = nn.DataParallel(s3_t_discriminator)
s3_t_discriminator=s3_t_discriminator.cuda()


def print_log(step, epoch, epoches, lr, l1, l2, l3, l4, l5, l6, l7, l8, l9,l10,l11,l12, flag, ploter, count):
    logger.info("Step [%d/%d] Epoch [%d/%d] lr: %f, s1_cls_loss: %.4f, s2_cls_loss: %.4f,s3_cls_loss: %.4f, s1_t_dis_loss: %.4f, " \
          "s2_t_dis_loss: %.4f, s3_t_dis_loss: %.4f, s1_t_confusion_loss_s1: %.4f, s1_t_confusion_loss_t: %.4f, " \
          "s2_t_confusion_loss_s2: %.4f, s2_t_confusion_loss_t: %.4f,s3_t_confusion_loss_s3: %.4f, s3_t_confusion_loss_t: %.4f, selected_source: %s" \
          % (step, steps, epoch, epoches, lr, l1, l2, l3, l4, l5, l6, l7, l8,l9,l10,l11,l12, flag))


count = 0
max_correct = 0
max_step = 0
max_epoch = 0
ploter = LinePlotter(env_name="bvlc_A_W_2_D")
for step in range(steps):
    # Part 1: assign psudo-labels to t-domain and update the label-dataset
    logger.info( "#################### Part1 ####################")
    extractor.eval()
    s1_classifier.eval()
    s2_classifier.eval()
    s3_classifier.eval()
    
    fin = open(t_label)
    fout = open(os.path.join(data_root, args.t, "pseudo/pse_label_" + str(step) + ".txt"), "w")
    if step > 0:
        s1_weight = s1_weight_loss / (s1_weight_loss + s2_weight_loss+ s3_weight_loss)
        s2_weight = s2_weight_loss / (s1_weight_loss + s2_weight_loss+ s3_weight_loss)
        s3_weight = s3_weight_loss / (s1_weight_loss + s2_weight_loss+ s3_weight_loss)
    logger.info( "s1_weight is:{}".format(s1_weight))
    logger.info( "s2_weight is:{}".format(s2_weight))
    logger.info("s3_weight is:{}".format(s3_weight))



    for i, (t_imgs, t_labels) in enumerate(t_loader_test):
        t_imgs = Variable(t_imgs.cuda())
        t_feature = extractor(t_imgs)
        s1_cls = s1_classifier(t_feature)
        s2_cls = s2_classifier(t_feature)
        s3_cls = s3_classifier(t_feature)
        s1_cls = F.softmax(s1_cls,dim=1)
        s2_cls = F.softmax(s2_cls,dim=1)
        s3_cls = F.softmax(s3_cls, dim=1)
        s1_cls = s1_cls.data.cpu().numpy()
        s2_cls = s2_cls.data.cpu().numpy()
        s3_cls = s3_cls.data.cpu().numpy()
        
        t_pred = s1_cls * s1_weight + s2_cls * s2_weight+ s3_cls * s3_weight
        #ids = t_pred.argmax(axis=1)
        ids=np.argmax(t_pred,axis=1)
        for j in range(ids.shape[0]):
            line = fin.next()
            data = line.strip().split(" ")
            if t_pred[j, ids[j]] >= threshold:
                fout.write(data[0] + " " + str(ids[j]) + "\n")


    fin.close()
    fout.close()     

  
    # Part 2: train F1t, F2t with pseudo labels
    logger.info( "#################### Part2 ####################")
    extractor.train()
    s1_classifier.train()
    s2_classifier.train()
    s3_classifier.train()
    t_pse_label = os.path.join(data_root, args.t, "pseudo/pse_label_" + str(step) + ".txt")
    t_pse_set = OfficeImage(t_root, t_pse_label, split="train")
    t_pse_loader_raw = torch.utils.data.DataLoader(t_pse_set, batch_size=batch_size,
                           shuffle=shuffle, num_workers=num_workers)
    logger.info( "Length of pseudo-label dataset:{}".format(len(t_pse_set)))

    optim_extract = optim.Adam(extractor.parameters(), lr=lr, betas=(beta1, beta2))
    optim_s1_cls = optim.Adam(s1_classifier.parameters(), lr=lr, betas=(beta1, beta2))
    optim_s2_cls = optim.Adam(s2_classifier.parameters(), lr=lr, betas=(beta1, beta2))
    optim_s3_cls = optim.Adam(s3_classifier.parameters(), lr=lr, betas=(beta1, beta2))

    for cls_epoch in range(cls_epoches):#cls_epoches
        s1_loader, s2_loader, s3_loader, t_pse_loader = iter(s1_loader_raw), iter(s2_loader_raw), iter(s3_loader_raw), iter(t_pse_loader_raw)
        for i, (t_pse_imgs, t_pse_labels) in enumerate(t_pse_loader):
            try:
                s1_imgs, s1_labels = s1_loader.next()
            except StopIteration:
                s1_loader = iter(s1_loader_raw)
                s1_imgs, s1_labels = s1_loader.next()
            try:
                s2_imgs, s2_labels = s2_loader.next()
            except StopIteration:
                s2_loader = iter(s2_loader_raw)
                s2_imgs, s2_labels = s2_loader.next()
            try:
                s3_imgs, s3_labels = s3_loader.next()
            except StopIteration:
                s3_loader = iter(s3_loader_raw)
                s3_imgs, s3_labels = s3_loader.next()
            s1_imgs, s1_labels = Variable(s1_imgs.cuda()), Variable(s1_labels.cuda())
            s2_imgs, s2_labels = Variable(s2_imgs.cuda()), Variable(s2_labels.cuda())
            s3_imgs, s3_labels = Variable(s3_imgs.cuda()), Variable(s3_labels.cuda())
            t_pse_imgs, t_pse_labels = Variable(t_pse_imgs.cuda()), Variable(t_pse_labels.cuda())
            
            s1_t_imgs = torch.cat((s1_imgs, t_pse_imgs), 0)
            s1_t_labels = torch.cat((s1_labels, t_pse_labels), 0)
            s2_t_imgs = torch.cat((s2_imgs, t_pse_imgs), 0)
            s2_t_labels = torch.cat((s2_labels, t_pse_labels), 0)
            s3_t_imgs = torch.cat((s3_imgs, t_pse_imgs), 0)
            s3_t_labels = torch.cat((s3_labels, t_pse_labels), 0)

            optim_extract.zero_grad()
            optim_s1_cls.zero_grad()
            optim_s2_cls.zero_grad()
            optim_s3_cls.zero_grad()

            s1_t_feature = extractor(s1_t_imgs)
            s2_t_feature = extractor(s2_t_imgs)
            s3_t_feature = extractor(s3_t_imgs)
            s1_t_cls = s1_classifier(s1_t_feature)
            s2_t_cls = s2_classifier(s2_t_feature)
            s3_t_cls = s3_classifier(s3_t_feature)
            s1_t_cls_loss = get_cls_loss(s1_t_cls, s1_t_labels)
            s2_t_cls_loss = get_cls_loss(s2_t_cls, s2_t_labels)
            s3_t_cls_loss = get_cls_loss(s3_t_cls, s3_t_labels)

            torch.autograd.backward([s1_t_cls_loss, s2_t_cls_loss, s3_t_cls_loss])

            optim_s1_cls.step()
            optim_s2_cls.step()
            optim_s3_cls.step()
            optim_extract.step()

            if (i+1) % log_interval == 0:
                print_log(step+1, cls_epoch+1, cls_epoches, lr, s1_t_cls_loss.data[0], \
                           s2_t_cls_loss.data[0],s3_t_cls_loss.data[0], 0, 0, 0, 0, 0, 0,0,0,0, "...", ploter, count)
                count += 1
    
        extractor.eval()
        s1_classifier.eval()
        s2_classifier.eval()
        s3_classifier.eval()
        correct = 0
        for (imgs, labels) in t_loader_test:
            imgs = Variable(imgs.cuda())
            imgs_feature = extractor(imgs)
            s1_cls = s1_classifier(imgs_feature)
            s2_cls = s2_classifier(imgs_feature)
            s3_cls = s3_classifier(imgs_feature)
            s1_cls = F.softmax(s1_cls,dim=1)
            s2_cls = F.softmax(s2_cls,dim=1)
            s3_cls = F.softmax(s3_cls, dim=1)
            s1_cls = s1_cls.data.cpu().numpy()
            s2_cls = s2_cls.data.cpu().numpy()
            s3_cls = s3_cls.data.cpu().numpy()
            res = s1_cls * s1_weight + s2_cls * s2_weight+ s3_cls * s3_weight
            #pred = res.argmax(axis=1)
            pred = np.argmax(res,axis=1)
            labels = labels.numpy()
            correct += np.equal(labels, pred).sum()
        current_accuracy = correct * 1.0 / len(t_set_test)
        current_accuracy=current_accuracy.item()
        logger.info( "Current accuracy is:{}".format(current_accuracy))

        if current_accuracy >= max_correct:
            max_correct = current_accuracy
            max_step = step
            max_epoch = cls_epoch
            torch.save(extractor.state_dict(), os.path.join(snapshot, "p2_extractor_" + str(step) + "_" + str(cls_epoch) + ".pth"))
            torch.save(s1_classifier.state_dict(), os.path.join(snapshot, os.path.abspath(__file__).
                                                                split('/')[-1].split('.')[0]+"p2_s1_cls_" + str(step) + "_" + str(cls_epoch) + ".pth"))
            torch.save(s2_classifier.state_dict(), os.path.join(snapshot, os.path.abspath(__file__).
                                                                split('/')[-1].split('.')[0]+"p2_s2_cls_" + str(step) + "_" + str(cls_epoch) + ".pth"))
            torch.save(s2_classifier.state_dict(), os.path.join(snapshot, os.path.abspath(__file__).
                                                                split('/')[-1].split('.')[0] + "p2_s3_cls_" + str(
                step) + "_" + str(cls_epoch) + ".pth"))
            
         
    # Part 3: train discriminator and generate mix feature
    logger.info( "#################### Part3 ####################")
    extractor.train()
    s1_classifier.train()
    s2_classifier.train()
    s3_classifier.train()
    optim_extract = optim.Adam(extractor.parameters(), lr=lr, betas=(beta1, beta2))
    optim_s1_t_dis = optim.Adam(s1_t_discriminator.parameters(), lr=lr, betas=(beta1, beta2))
    optim_s2_t_dis = optim.Adam(s2_t_discriminator.parameters(), lr=lr, betas=(beta1, beta2))
    optim_s3_t_dis = optim.Adam(s3_t_discriminator.parameters(), lr=lr, betas=(beta1, beta2))

    s1_weight_loss = 0
    s2_weight_loss = 0
    s3_weight_loss = 0
    for gan_epoch in range(gan_epoches):#gan_epoches
        s1_loader, s2_loader,s3_loader, t_loader = iter(s1_loader_raw), iter(s2_loader_raw), iter(s3_loader_raw), iter(t_loader_raw)
        for i, (t_imgs, t_labels) in enumerate(t_loader):
            s1_imgs, s1_labels = s1_loader.next()
            s2_imgs, s2_labels = s2_loader.next()
            s3_imgs, s3_labels = s3_loader.next()
            s1_imgs, s1_labels = Variable(s1_imgs.cuda()), Variable(s1_labels.cuda())
            s2_imgs, s2_labels = Variable(s2_imgs.cuda()), Variable(s2_labels.cuda())
            s3_imgs, s3_labels = Variable(s3_imgs.cuda()), Variable(s3_labels.cuda())
            t_imgs = Variable(t_imgs.cuda())

            #train G
            optim_extract.zero_grad()
            s1_feature = extractor(s1_imgs)
            s2_feature = extractor(s2_imgs)
            s3_feature = extractor(s3_imgs)
            t_feature = extractor(t_imgs)
            s1_cls = s1_classifier(s1_feature)
            s2_cls = s2_classifier(s2_feature)
            s3_cls = s2_classifier(s3_feature)
            s1_t_fake = s1_t_discriminator(s1_feature)
            s1_t_real = s1_t_discriminator(t_feature)
            s2_t_fake = s2_t_discriminator(s2_feature)
            s2_t_real = s2_t_discriminator(t_feature)
            s3_t_fake = s3_t_discriminator(s3_feature)
            s3_t_real = s3_t_discriminator(t_feature)

            s1_cls_loss = get_cls_loss(s1_cls, s1_labels)
            s2_cls_loss = get_cls_loss(s2_cls, s2_labels)
            s3_cls_loss = get_cls_loss(s3_cls, s3_labels)
            s1_t_dis_loss = get_dis_loss(s1_t_fake, s1_t_real)
            s2_t_dis_loss = get_dis_loss(s2_t_fake, s2_t_real)
            s3_t_dis_loss = get_dis_loss(s3_t_fake, s3_t_real)
            s1_weight_loss += s1_t_dis_loss.data[0]
            s2_weight_loss += s2_t_dis_loss.data[0]
            s3_weight_loss += s3_t_dis_loss.data[0]

            s1_t_confusion_loss_s1 = get_confusion_loss(s1_t_fake)
            s1_t_confusion_loss_t = get_confusion_loss(s1_t_real)            
            s1_t_confusion_loss = 0.5 * s1_t_confusion_loss_s1 + 0.5 * s1_t_confusion_loss_t

            s2_t_confusion_loss_s2 = get_confusion_loss(s2_t_fake)
            s2_t_confusion_loss_t = get_confusion_loss(s2_t_real)
            s2_t_confusion_loss = 0.5 * s2_t_confusion_loss_s2 + 0.5 * s2_t_confusion_loss_t

            s3_t_confusion_loss_s3 = get_confusion_loss(s3_t_fake)
            s3_t_confusion_loss_t = get_confusion_loss(s3_t_real)
            s3_t_confusion_loss = 0.5 * s3_t_confusion_loss_s3 + 0.5 * s3_t_confusion_loss_t

            if s1_t_dis_loss.data[0] > s2_t_dis_loss.data[0] and s1_t_dis_loss.data[0] > s3_t_dis_loss.data[0]:
                SELECTIVE_SOURCE = "S1"
                torch.autograd.backward([s1_cls_loss, s2_cls_loss,s3_cls_loss, s1_t_confusion_loss])
            elif s1_t_dis_loss.data[0] < s2_t_dis_loss.data[0] and s2_t_dis_loss.data[0] > s3_t_dis_loss.data[0]:
                SELECTIVE_SOURCE = "S2"
                torch.autograd.backward([s1_cls_loss, s2_cls_loss,s3_cls_loss, s2_t_confusion_loss])
            else:
                SELECTIVE_SOURCE = "S3"
                torch.autograd.backward([s1_cls_loss, s2_cls_loss,s3_cls_loss, s3_t_confusion_loss])

            optim_extract.step()

            #train D
            s1_t_discriminator.zero_grad()
            s2_t_discriminator.zero_grad()
            s3_t_discriminator.zero_grad()
            s1_t_fake = s1_t_discriminator(s1_feature.detach())
            s1_t_real = s1_t_discriminator(t_feature.detach())
            s2_t_fake = s2_t_discriminator(s2_feature.detach())
            s2_t_real = s2_t_discriminator(t_feature.detach())
            s3_t_fake = s3_t_discriminator(s3_feature.detach())
            s3_t_real = s3_t_discriminator(t_feature.detach())
            s1_t_dis_loss = get_dis_loss(s1_t_fake, s1_t_real)
            s2_t_dis_loss = get_dis_loss(s2_t_fake, s2_t_real)
            s3_t_dis_loss = get_dis_loss(s3_t_fake, s3_t_real)
            torch.autograd.backward([s1_t_dis_loss, s2_t_dis_loss, s3_t_dis_loss])
            optim_s1_t_dis.step()
            optim_s2_t_dis.step()
            optim_s3_t_dis.step()

            if (i+1) % log_interval == 0:
                print_log(step+1, gan_epoch+1, gan_epoches, lr, s1_cls_loss.data[0], s2_cls_loss.data[0], s3_cls_loss.data[0], s1_t_dis_loss.data[0], \
                          s2_t_dis_loss.data[0],s3_t_dis_loss.data[0], s1_t_confusion_loss_s1.data[0], s1_t_confusion_loss_t.data[0], \
                          s2_t_confusion_loss_s2.data[0], s2_t_confusion_loss_t.data[0],s3_t_confusion_loss_s3.data[0], s3_t_confusion_loss_t.data[0], SELECTIVE_SOURCE, ploter, count)
                count += 1

logger.info("max_correct is :{}".format(str(max_correct)))
logger.info("max_step is :{}".format(str(max_step+1)))
logger.info("max_epoch is :{}".format(str(max_epoch+1)))
