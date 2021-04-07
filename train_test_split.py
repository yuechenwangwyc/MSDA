import os
import os.path as osp
import numpy as np

'''
MAIN_DIR=os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
f_train = open(osp.join(MAIN_DIR,"dataset/office-home/train_test","Art_train.txt"), "w")
f_test = open(osp.join(MAIN_DIR,"dataset/office-home/train_test","Art_test.txt"), "w")
labeltxt = open(osp.join(MAIN_DIR,"dataset/office-home","Art.txt"))
i=0
data_list=[]
data_l=[]
for line in labeltxt:
    data_list.append(line)

ridx=np.arange(len(data_list))
np.random.shuffle(ridx)

for num in ridx:
    data_l.append(data_list[num])



for line in data_l:
    i=i+1
    if i<=2000:
        f_train.write(line)
    if i>2000:
        f_test.write(line)

'''

MAIN_DIR=os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
f_train = open(osp.join(MAIN_DIR,"dataset/office-home/train_test","Product_train1.txt"), "w")
f_test = open(osp.join(MAIN_DIR,"dataset/office-home/train_test","Product_test1.txt"), "w")
labeltxt = open(osp.join(MAIN_DIR,"dataset/office-home","Real_World.txt"))
i=0
data_list=[]
data_l=[]
for line in labeltxt:
    data_list.append(line)

ridx=np.arange(len(data_list))
np.random.shuffle(ridx)

for num in ridx:
    data_l.append(data_list[num])



for line in data_l:
    i=i+1
    if i<=4000:
        f_train.write(line)
    if i>4000:
        f_test.write(line)