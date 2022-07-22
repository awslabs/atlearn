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

from ATLearn.algorithms.BaseTL import BaseTL
import torch
import time
import glob
import os
from datetime import datetime
from ATLearn.utils.data_loader import read_data_from_folder
from tqdm import tqdm


class DTL_image_classification(BaseTL):
    def __init__(self, data, user_network=None, network='resnet50', retrain=False, freeze=True, gpu_id=-1,
                 labeled_data=True, epochs=10, lr=1e-3, batch_size=32, save_every_epoch=500):
        '''
        A dynamic transfer learning for image classification
        :param data: path to load the training examples.
            A folder includes multiple sub-folders where each sub-folder contains the training examples in one class
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
        super(DTL_image_classification, self).__init__(data, user_network, network, retrain, freeze, gpu_id)
        self.base_network_name = network
        self.labeled_data = labeled_data
        self.epochs = epochs
        self.save_every_epoch = save_every_epoch
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_loader, num_classes = read_data_from_folder(self.data, batch_size, mode='train')
        if not retrain:
            self.fcn = torch.nn.Linear(self.fts_dim, num_classes).to(device=self.device)
        self.get_optimizer(lr)

        old_models = "../../save"
        if not os.path.exists(old_models):
            os.makedirs(old_models)
        self.old_networks = []
        self.time_stamps = []
        if os.listdir(old_models):
            model_names = glob.glob(old_models + '/*.pt')
            for name in model_names:
                if "params" not in name:
                    self.old_networks.append(torch.jit.load(name))
                    self.time_stamps.append(name[-13:-3])

    def train_model(self):
        date = datetime.now().strftime("%Y_%m_%d")
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
                loss = 0.
                if self.labeled_data:
                    loss += self.criterion(preds, labels)

                decay_rate = 0.8
                if len(self.old_networks) > 0:
                    for k, old_model in reversed(list(enumerate(self.old_networks))):
                        ratio = self.estimate_decay(self.time_stamps[k], date)
                        pseudo_logits = old_model(images)
                        _, pseudo_labels = torch.max(pseudo_logits, dim=-1)
                        loss += self.criterion(preds, pseudo_labels) * (decay_rate ** ratio)

                loss.backward()
                self.optimizer.step()
                training_loss += loss

            if epoch % self.save_every_epoch == 0:
                self.save_checkpoint(epoch, training_loss)
        print("Model training is done with {:.4f} seconds!".format(time.time() - t_start))

    @staticmethod
    def estimate_decay(old, new):
        year, month, day = old.split('_')
        old = datetime(int(year), int(month), int(day)).timestamp()
        year, month, day = new.split('_')
        new = datetime(int(year), int(month), int(day)).timestamp()
        return new - old


if __name__ == '__main__':
    model = DTL_image_classification(data="/Users/junuiuc/Downloads/DTL/data/Fruit/train/")
    model.train_model()
    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    model.export(save_name="banana_pu", save_path="../../save")
    # model.predict(input_data="/Users/junuiuc/Desktop/test_sample.png")




