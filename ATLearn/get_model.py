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

from ATLearn.algorithms.TL_image_classification import TL_image_classification
from ATLearn.algorithms.TL_image_classification_PU import TL_image_classification_PU
from ATLearn.algorithms.TL_image_classification_OC import TL_image_classification_OC
from ATLearn.algorithms.TL_regression import TL_regression
from ATLearn.algorithms.TL_object_detection import TL_object_detection


def get_model(mode, algorithm, data,
              val_data=None,
              user_network=None,
              network=None,
              retrain=False,
              freeze=True,
              gpu_id=-1,
              options=None):
    model = None
    if mode == "image_classification":
        assert algorithm in ['TL_image_classification', 'TL_image_classification_PU', 'TL_image_classification_OC',
                             'TL_regression']
        if network is None:
            network = 'resnet50'
        instance_gen = globals()[algorithm]
        model = instance_gen(data=data,
                             user_network=user_network,
                             network=network,
                             retrain=retrain,
                             freeze=freeze,
                             gpu_id=gpu_id,
                             options=options)
    elif mode == "object_detection":
        assert algorithm in ['TL_object_detection']
        if network is None:
            network = 'yolov5s'
        instance_gen = globals()[algorithm]
        model = instance_gen(data=data,
                             val_data=val_data,
                             user_network=user_network,
                             network=network,
                             freeze=freeze,
                             gpu_id=gpu_id,
                             options=options)
    else:
        print("This task is not implemented yet.")

    return model
