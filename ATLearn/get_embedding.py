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

def get_embedding(task,
                  save_path,
                  user_network=None,
                  network='resnet50',
                  retrain=False,
                  freeze=True):
    model = None
    if task == "image_classification":
        if user_network is None:
            instance_gen = globals()[network]
            base_network = instance_gen(pretrained=True)
        else:
            if retrain:
                base_network = torch.jit.load(user_network)
            else:
                base_network = torch.load(user_network)

        if 'vit' not in network:
            base_network = torch.nn.Sequential(*list(base_network.children())[:-1])
            base_network.append(torch.nn.Flatten())

        for param in base_network.parameters():
            param.requires_grad = not freeze

        # export embedding
        base_network.train(False)
        test_input = torch.rand(1, 3, 224, 224)
        traced_base_network = torch.jit.trace(base_network, test_input)
        traced_base_network.save(save_path + "/" + network + "_embedding.pt")

    else:
        print("This task is not implemented yet.")

    return model
