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
import torch.nn.functional as F
import time
from tqdm import tqdm
from ATLearn.utils.data_loader import read_data_from_folder_pu


class TL_image_classification_PU(BaseTL):
    def __init__(self, data, user_network=None, network='resnet50', retrain=False, freeze=True, gpu_id=-1,
                 options=None, beta=0.5, rho=1, warmup_epochs=50, epochs=100, lr=1e-3, batch_size=32,
                 save_every_epoch=10):
        '''
        A PU-based transfer learning for image classification with positive samples and unlabeled samples
        :param data: path to load the training examples (default: the first folder contains the positive examples)
        :param user_network: customers' own pre-trained model
        :param network: a large pre-trained network
        :param retrain: whether to retrain the model. If so, model architecture of pre-trained model will not be changed.
        :param freeze: whether to freeze the pre-trained layers
        :param gpu_id: whether to use GPUs
        :param beta: hyper-parameter of positive-unlabeled learning, i.e., class-prior probability of negative class
        :param rho: trade-off of pu loss and pseudo-labeling loss
        :param warmup_epochs: pseudo-labeling is activated after several warmup epochs
        :param epochs: total number of training epochs
        :param lr: learning rate
        :param batch_size: batch size
        :param save_every_epoch: save checkpoint at some steps
        '''
        super(TL_image_classification_PU, self).__init__(data, user_network, network, retrain, freeze, gpu_id, options)
        self.base_network_name = network
        self.beta = beta
        self.rho = rho
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.save_every_epoch = save_every_epoch
        self.train_loader, self.num_classes = read_data_from_folder_pu(self.data, batch_size, mode='train')
        if not retrain:
            self.fcn = torch.nn.Linear(self.fts_dim, self.num_classes).to(device=self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.get_optimizer(lr)

    def train_model(self):
        '''
        Positive-unlabeled learning and pseudo-labeling are adopted here.
        * Kiryo, Ryuichi, et al. "Positive-unlabeled learning with non-negative risk estimator." In NeurIPS. 2017.
        * Lee, Dong-Hyun. "Pseudo-label: The simple and efficient semi-supervised learning method for deep neural
                           networks." In ICML Workshop on challenges in representation learning. 2013.
        '''
        train_loader_p, train_loader_u = self.train_loader
        num_batch = min(len(train_loader_p), len(train_loader_u))
        t_start = time.time()
        for epoch in tqdm(range(1, self.epochs+1)):
            self.base_network.train()
            self.fcn.train()

            training_loss = 0.
            iter_train_p, iter_train_u = iter(train_loader_p), iter(train_loader_u)
            idx_unknown = self.num_classes - 1
            for _ in range(num_batch):
                self.optimizer.zero_grad()
                images_p, labels_p = next(iter_train_p)
                images_u, labels_u = next(iter_train_u)
                images_p, labels_p = images_p.to(self.device), labels_p.to(self.device)
                images_u, labels_u = images_u.to(self.device), labels_u.to(self.device)

                images = torch.cat([images_p, images_u], dim=0)
                labels = torch.cat([labels_p, labels_u], dim=0)
                feats = self.base_network(images)
                preds = self.fcn(feats)

                p_loss = self.criterion(preds[labels != idx_unknown], labels[labels != idx_unknown]) * self.beta
                u_loss = self.criterion(preds[labels == idx_unknown], labels[labels == idx_unknown])
                u_labels = torch.ones_like(labels[labels != idx_unknown]) * idx_unknown
                u_loss -= self.criterion(preds[labels != idx_unknown], u_labels) * self.beta
                loss = p_loss + torch.nn.functional.leaky_relu(u_loss, negative_slope=1e-9)

                if epoch > self.warmup_epochs:
                    T = 2
                    threshold = 0.95
                    logits_u = preds[labels == idx_unknown]
                    pseudo_label = torch.softmax(logits_u.detach() / T, dim=-1)
                    max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                    mask = max_probs.ge(threshold).float()
                    loss += (F.cross_entropy(logits_u, targets_u, reduction='none') * mask).mean() * self.rho

                loss.backward()
                self.optimizer.step()
                training_loss += loss

            if epoch % self.save_every_epoch == 0:
                self.save_checkpoint(epoch, training_loss)
        print("Model training is done with {:.4f} seconds!".format(time.time() - t_start))






