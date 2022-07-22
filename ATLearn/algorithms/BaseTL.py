# Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
# https://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

from ATLearn.model_zoo.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from ATLearn.model_zoo.alexnet import alexnet
from ATLearn.model_zoo.vgg import vgg11, vgg13, vgg16, vgg19
from ATLearn.model_zoo.densenet import densenet121, densenet161, densenet169, densenet201
from ATLearn.model_zoo.squeezenet import squeezenet1_0, squeezenet1_1
from ATLearn.model_zoo.vit import vit_b_16, vit_b_32, vit_l_16, vit_l_32
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


class BaseTL(nn.Module):
    def __init__(self, data, user_network=None, network='resnet50', retrain=False, freeze=True, gpu_id=-1):
        super(BaseTL, self).__init__()
        if gpu_id >= 0 and torch.cuda.is_available():
            self.device = torch.device("cuda:{}".format(gpu_id))
        elif gpu_id >= 0 and not torch.cuda.is_available():
            self.device = torch.device("cpu")
            print("No GPU is found! It will use CPU directly.")
        else:
            self.device = torch.device("cpu")
        self.data = data

        if user_network is None:
            instance_gen = globals()[network]
            self.base_network = instance_gen(pretrained=True)
        else:
            if retrain:
                self.base_network = torch.jit.load(user_network)
            else:
                self.base_network = torch.load(user_network)
        if 'vit' not in network:
            self.base_network = torch.nn.Sequential(*list(self.base_network.children())[:-1])
            self.base_network.append(torch.nn.Flatten())
        self.base_network = self.base_network.to(device=self.device)
        d = self.base_network(torch.rand(1, 3, 224, 224).to(device=self.device))
        self.fts_dim = d.shape[1]
        for param in self.base_network.parameters():
            param.requires_grad = freeze
        if retrain:
            self.fcn = torch.nn.Sequential(*list(self.base_network.children())[-1:])
            print(self.fcn)
        else:
            self.fcn = None
        self.optimizer = None
        self.base_names, self.classifier_names = [], []
        self.save_traced_network = None

    def forward(self, inputs):
        inputs = self.cnn(inputs)
        inputs = self.fcn(inputs)
        return inputs

    def get_optimizer(self, lr):
        params_to_update = []
        for name, param in self.base_network.named_parameters():
            self.base_names.append(name)
            if param.requires_grad:
                params_to_update.append({'params': param, 'lr': 0.1 * lr})
        for name, param in self.fcn.named_parameters():
            self.classifier_names.append(name)
            if param.requires_grad:
                params_to_update.append({'params': param, 'lr': lr})
        self.optimizer = optim.Adam(params_to_update, weight_decay=5e-4)

    def train_model(self):
        raise NotImplementedError

    def mixup(self, inputs, labels, alpha=1.0):
        '''
        A simple data augmentation approach from
        * Zhang, Hongyi, et al. "mixup: Beyond Empirical Risk Minimization." In ICLR. 2018.
        :param inputs: input images (input-level mixup)
        :param alpha: a hyper-parameter for generating the mixture magnitude
        :return: mixed images, and generated mixup factor
        '''
        lam = np.random.beta(alpha, alpha)
        index = torch.randperm(inputs.shape[0])
        inputs_mix = lam * inputs + (1 - lam) * inputs[index]
        return inputs_mix, lam, labels, labels[index]

    def export(self, save_name="model", save_path="../"):
        date = datetime.now().strftime("%Y_%m_%d")
        model = torch.nn.Sequential()
        model.append(self.base_network)
        model.append(self.fcn)
        self.save_traced_network = torch.jit.trace(model.eval(), torch.rand(1, 3, 224, 224).to(self.device))  # .eval() is required here
        self.save_traced_network.save('{}/{}_{}.pt'.format(save_path, save_name, date))
        torch.save({
            'base_params_names': self.base_names,
            'detection_params_names': self.classifier_names
        }, '{}/{}_params_{}.pt'.format(save_path, save_name, date))
        self.print_text("TL model has been traced, and saved at {}/{}_{}.pt".format(save_path, save_name, date))

    def save_checkpoint(self, epoch, training_loss):
        date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
        model = torch.nn.Sequential()
        model.append(self.base_network)
        model.append(self.fcn)
        model.eval()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': training_loss,
        }, "../checkpoint/checkpoint_epoch_{}_{}.pt".format(epoch, date))

    def predict_regression(self, input_data, show=True):
        trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()])
        image = Image.open(input_data).convert('RGB')
        plt.imshow(image, cmap=plt.cm.binary)
        image = trans(image).float()
        image = image.unsqueeze(0)
        self.save_traced_network.eval()
        with torch.no_grad():
            logits = self.save_traced_network(image)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("Predicted score: {}".format(
            logits[0].item()),
            color='red')
        if show:
            plt.show()

    def predict(self, input_data, class_names=None, show=True):
        # self.save_traced_network = torch.jit.load('../banana_pu.pt').to(self.device)
        trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()])
        image = Image.open(input_data).convert('RGB')
        plt.imshow(image, cmap=plt.cm.binary)
        image = trans(image).float()
        image = image.unsqueeze(0)
        self.save_traced_network.eval()
        with torch.no_grad():
            logits = self.save_traced_network(image)
        ps = torch.exp(logits)
        ps = ps / torch.sum(ps)
        _, predIndex = torch.max(ps, 1)
        plt.xticks([])
        plt.yticks([])
        if class_names is None:
            class_names = np.arange(logits.shape[1])
        plt.xlabel("Predicted Class Label: {}, Confidence: {:0.2f}%".format(
            class_names[predIndex.item()], 100 * np.max(ps.numpy())),
            color='red')
        if show:
            plt.show()
        return class_names[predIndex.item()]

    @staticmethod
    def print_text(text):
        print("\033[91m {}\033[00m".format(text))

    @staticmethod
    def set_random_seed(seed=0):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
