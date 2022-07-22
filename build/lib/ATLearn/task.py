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

class task_manager(object):
    def __init__(self):
        self.IMAGE_CLASSIFICATION = 'image_classification'
        self.OBJECT_DETECTION = 'object_detection'


class algorithm_manager(object):
    def __init__(self):
        self.IC_STANDARD_TRANSFER = 'TL_image_classification'
        self.REGRESSION = 'TL_regression'
        self.IC_POSITIVE_UNLABELED = 'TL_image_classification_PU'
        self.IC_ONE_CLASS = 'TL_image_classification_OC'
        self.OD_STANDARD_TRANSFER = 'TL_object_detection'
