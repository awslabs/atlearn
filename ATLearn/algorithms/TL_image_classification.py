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

from .BaseTL import BaseTL
import torch
import time
from ATLearn.utils.data_loader import read_data_from_folder
from tqdm import tqdm


class TL_image_classification(BaseTL):
    def __init__(self, data, user_network=None, network='resnet50', retrain=False, freeze=True, gpu_id=-1,
                 options=None, aug=False, rho=0.1, epochs=10, lr=1e-3, batch_size=32, save_every_epoch=5):
        '''
        A standard transfer learning for image classification
        :param data: path to load the training examples.
            A folder includes multiple sub-folders where each sub-folder contains the training examples in one class:
            path/class_a/1.png
            path/class_a/2.png
            ...
            path/class_b/1.png
            path/class_b/2.png
            ...
            path/class_c/1.png
            path/class_c/2.png
            ...
        :param user_network: customers' own pre-trained model
        :param network: a large pre-trained network
        :param retrain: whether to retrain the model. If so, model architecture of pre-trained model will not be changed.
        :param freeze: whether to freeze the pre-trained layers
        :param gpu_id: whether to use GPUs
        :param aug: whether to use mixup for data augmentation
        :param rho: trade-off of raw examples and augmented examples
        :param epochs: total number of training epochs
        :param lr: learning rate
        :param batch_size: batch size
        :param save_every_epoch: save checkpoint at some steps
        '''
        super(TL_image_classification, self).__init__(data, user_network, network, retrain, freeze, gpu_id, options)
        self.base_network_name = network
        self.aug = aug
        self.rho = rho
        self.epochs = epochs
        self.save_every_epoch = save_every_epoch
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_loader, num_classes = read_data_from_folder(self.data, batch_size, mode='train')
        if not retrain:
            self.fcn = torch.nn.Linear(self.fts_dim, num_classes).to(device=self.device)
        self.get_optimizer(lr)

    def train_model(self):
        t_start = time.time()
        for epoch in tqdm(range(1, self.epochs+1)):
            self.base_network.train()
            self.fcn.train()

            training_loss = 0.
            for (images, labels) in self.train_loader:
                self.optimizer.zero_grad()
                images, labels = images.to(self.device), labels.to(self.device)
                feats = self.base_network(images)
                preds = self.fcn(feats)
                loss = self.criterion(preds, labels)

                if self.aug:
                    images_mix, lam, labels_o, labels_i = self.mixup(images, labels)
                    feats_mix = self.base_network(images_mix)
                    preds_mix = self.fcn(feats_mix)
                    loss_mix = self.criterion(preds_mix, labels_o) * lam + self.criterion(preds_mix, labels_i) * (1-lam)
                    loss += loss_mix * self.rho

                loss.backward()
                self.optimizer.step()
                training_loss += loss

            if epoch % self.save_every_epoch == 0:
                self.save_checkpoint(epoch, training_loss)
        print("Model training is done with {:.4f} seconds!".format(time.time() - t_start))






