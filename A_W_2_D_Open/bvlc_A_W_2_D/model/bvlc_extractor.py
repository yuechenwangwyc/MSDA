import torch
import torch.nn as nn
from torchvision import models



def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform(m.weight)
        nn.init.constant(m.bias,0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal(m.weight, 1.0, 0.02)
        nn.init.constant(m.bias,0.0)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight)
        nn.init.constant(m.bias,0.0)



resnet_dict = {"ResNet18":models.resnet18, "ResNet34":models.resnet34, "ResNet50":models.resnet50, "ResNet101":models.resnet101, "ResNet152":models.resnet152}
class ResNetFc(nn.Module):
  def __init__(self, resnet_name, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=1000):
    super(ResNetFc, self).__init__()
    model_resnet = resnet_dict[resnet_name](pretrained=True)
    self.conv1 = model_resnet.conv1
    self.bn1 = model_resnet.bn1
    self.relu = model_resnet.relu
    self.maxpool = model_resnet.maxpool
    self.layer1 = model_resnet.layer1
    self.layer2 = model_resnet.layer2
    self.layer3 = model_resnet.layer3
    self.layer4 = model_resnet.layer4
    self.avgpool = model_resnet.avgpool
    self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                         self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
    self.select_layers = nn.Sequential(self.layer3, self.layer4, self.avgpool)

    self.use_bottleneck = use_bottleneck
    self.sigmoid = nn.Sigmoid()
    self.new_cls = new_cls
    if new_cls:
        if self.use_bottleneck:
            self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.bottleneck.apply(init_weights)
            self.fc.apply(init_weights)

            self.__in_features = bottleneck_dim
        else:
            self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
            self.fc.apply(init_weights)
            self.gvbg = nn.Linear(model_resnet.fc.in_features, class_num)
            self.gvbg.apply(init_weights)
            self.__in_features = model_resnet.fc.in_features
    else:
        self.fc = model_resnet.fc
        self.__in_features = model_resnet.fc.in_features

  def forward(self, x, gvbg=True):
    x = self.feature_layers(x)
    x = x.view(x.size(0), -1)
    if self.use_bottleneck and self.new_cls:
        x = self.bottleneck(x)
    bridge = x
    y = self.fc(x)

    return x, y, bridge

  def output_num(self):
    return self.__in_features

  def get_parameters(self):
    if self.new_cls:
        if self.use_bottleneck:

            parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
        else:

            parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2},
                            {"params":self.gvbg.parameters(), "lr_mult":10, 'decay_mult':2}]
    else:
        parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
    return parameter_list








class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
            # self.average = nn.functional.avg_pool3d(kernel_size=(local_size, 1, 1),
            #                                         stride=1,
            #                                         padding=(int((local_size - 1.0) / 2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x


class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.norm1 = LRN(local_size=5, alpha=0.0001, beta=0.75)
  
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.norm2 = LRN(local_size=5, alpha=0.0001, beta=0.75)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
    
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout=nn.Dropout(p=0.1)

    def forward(self, input):
        dropout=self.dropout
        input=dropout(input)
        conv1 = self.conv1(input)
        relu1 = self.relu1(conv1)
        pool1 = self.pool1(relu1)
        norm1 = self.norm1(pool1)

        norm1 = dropout(norm1)
        conv2 = self.conv2(norm1)
        relu2 = self.relu2(conv2)
        pool2 = self.pool2(relu2)
        norm2 = self.norm2(pool2)

        norm2 = dropout(norm2)
        conv3 = self.conv3(norm2)
        relu3 = self.relu3(conv3)

        relu3 = dropout(relu3)
        conv4 = self.conv4(relu3)
        relu4 = self.relu4(conv4)

        relu4 = dropout(relu4)
        conv5 = self.conv5(relu4)
        relu5 = self.relu5(conv5)
        pool5 = self.pool5(relu5)
        return pool5


class Extractor1(nn.Module):
    def __init__(self):
        super(Extractor1, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.norm1 = LRN(local_size=5, alpha=0.0001, beta=0.75)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.norm2 = LRN(local_size=5, alpha=0.0001, beta=0.75)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, input):
        dropout = self.dropout

        conv1 = self.conv1(input)
        relu1 = self.relu1(conv1)
        pool1 = self.pool1(relu1)
        norm1 = self.norm1(pool1)


        conv2 = self.conv2(norm1)
        relu2 = self.relu2(conv2)
        pool2 = self.pool2(relu2)
        norm2 = self.norm2(pool2)


        conv3 = self.conv3(norm2)
        relu3 = self.relu3(conv3)


        conv4 = self.conv4(relu3)
        relu4 = self.relu4(conv4)


        conv5 = self.conv5(relu4)
        relu5 = self.relu5(conv5)
        pool5 = self.pool5(relu5)
        return pool5
