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

# readme

from .BaseTL import BaseTL
import torch
import time
from ATLearn.utils.data_loader import read_data_from_folder_regression
from tqdm import tqdm


class TL_regression(BaseTL):
    def __init__(self, data, user_network=None, network='resnet50', freeze=True, gpu_id=-1,
                 epochs=10, lr=1e-3, batch_size=32, save_every_epoch=5):
        super(TL_regression, self).__init__(data, user_network, network, freeze, gpu_id)
        '''
        A standard transfer learning for image regression
        :param path: path to load the training examples  
        :param user_network: customers' own base model
        :param network: a large pre-trained network
        :param freeze: whether to freeze the pre-trained layers
        :param gpu_id: whether to use GPUs
        :param epochs: total number of training epochs
        :param lr: learning rate
        :param batch_size: batch size
        :param save_every_epoch: save checkpoint at some steps
        '''
        self.base_network_name = network
        self.epochs = epochs
        self.save_every_epoch = save_every_epoch
        self.criterion = torch.nn.MSELoss()
        self.train_loader = read_data_from_folder_regression(self.data, batch_size, mode='train')
        self.fcn = torch.nn.Linear(self.fts_dim, 1).to(device=self.device)
        self.get_optimizer(lr)

    def train_model(self):
        t_start = time.time()
        for epoch in tqdm(range(1, self.epochs+1)):
            self.base_network.train()
            self.fcn.train()

            training_loss = 0.
            for (images, labels) in self.train_loader:
                self.optimizer.zero_grad()
                images, labels = images.to(self.device), labels.float().to(self.device)
                feats = self.base_network(images)
                preds = self.fcn(feats)
                loss = self.criterion(preds.flatten(), labels)

                loss.backward()
                self.optimizer.step()
                training_loss += loss

            if epoch % self.save_every_epoch == 0:
                self.save_checkpoint(epoch, training_loss)
        print("Model training is done with {:.4f} seconds!".format(time.time() - t_start))






