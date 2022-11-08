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
from ATLearn.utils.data_loader import read_data_from_folder_oc
from tqdm import tqdm


class TL_image_classification_OC(BaseTL):
    def __init__(self, data, user_network=None, network='resnet50', retrain=False, freeze=True, gpu_id=-1, options=None,
                 method="PseudoPU", beta=0.6, dist="euclidean", epochs=20, lr=1e-3, batch_size=32, save_every_epoch=5):
        '''
        Deep one-class transfer learning for image classification with only positive samples
        :param data: path to load the training examples
        :param user_network: customers' own pre-trained model
        :param network: a large pre-trained network
        :param retrain: whether to retrain the model. If so, model architecture of pre-trained model will not be changed.
        :param freeze: whether to freeze the pre-trained layers
        :param gpu_id: whether to use GPUs
        :param method: one-class classification method, including PseudoPU and SVDD
        :param beta: hyper-parameter of positive-unlabeled learning
        :param dist: which similarity/distnace metric to use in PseudoPU, including euclidean and cosine
        :param epochs: total number of training epochs
        :param lr: learning rate
        :param batch_size: batch size
        :param save_every_epoch: save checkpoint at some steps
        '''
        super(TL_image_classification_OC, self).__init__(data, user_network, network, retrain, freeze, gpu_id, options)
        self.base_network_name = network
        self.method = method
        self.beta = beta
        self.dist = dist
        self.epochs = epochs
        self.save_every_epoch = save_every_epoch
        self.train_loader = read_data_from_folder_oc(self.data, batch_size, mode='train')
        if not retrain:
            self.fcn = torch.nn.Linear(self.fts_dim, 128).to(device=self.device)
        self.get_optimizer(lr)

    def train_model(self):
        '''
        Methods:
            SVDD: Ruff, Lukas, et al. "Deep one-class classification." In ICML. 2018.
            DROCC: Goyal, Sachin, et al. "DROCC: Deep robust one-class classification." In ICML. 2020.
            Pseudo-PU: A pu-learning based one-class classification approach.
        '''
        t_start = time.time()
        for epoch in tqdm(range(1, self.epochs+1)):
            self.base_network.train()
            self.fcn.train()

            training_loss = 0.
            c0 = 0  # center of samples from one class
            c1 = 1
            for ((images_real, images_syn), labels) in self.train_loader:
                self.optimizer.zero_grad()
                images_real, images_syn = images_real.to(self.device), images_syn.to(self.device)

                loss = 0
                if self.method == "PseudoPU":
                    images = torch.cat([images_real, images_syn], dim=0)
                    feats = self.base_network(images)
                    preds = self.fcn(feats)
                    preds_real, preds_syn = preds.chunk(2)
                    if self.dist == "euclidean":
                        p_loss = torch.mean(torch.mean((preds_real - c0) ** 2, dim=1)) * self.beta
                        u_loss = torch.mean(torch.mean((preds_syn - c1) ** 2, dim=1))
                        u_loss -= torch.mean(torch.mean((preds_real - c1) ** 2, dim=1)) * self.beta
                        loss = p_loss + torch.nn.functional.leaky_relu(u_loss, negative_slope=1e-9)
                    elif self.dist == "cosine":
                        c0 = torch.ones(preds_real.shape[1])
                        c1 = - torch.ones(preds_real.shape[1])
                        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                        p_loss = torch.mean(torch.exp(-cos(preds_real, c0))) * self.beta
                        u_loss = torch.mean(torch.exp(-cos(preds_syn, c1)))
                        u_loss -= torch.mean(torch.exp(-cos(preds_real, c1))) * self.beta
                        loss = p_loss + torch.nn.functional.leaky_relu(u_loss, negative_slope=1e-9)

                elif self.method == "SVDD":
                    feats_real = self.base_network(images_real)
                    preds_real = self.fcn(feats_real)
                    loss = torch.mean(torch.mean((preds_real - c0) ** 2, dim=1))

                else:
                    print("Unknown method!")

                loss.backward()
                self.optimizer.step()
                training_loss += loss

            if epoch % self.save_every_epoch == 0:
                self.save_checkpoint(epoch, training_loss)
        print("Model training is done with {:.4f} seconds!".format(time.time() - t_start))






