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

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def accuracy(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        count = 0
        for (x, y) in test_loader:
            correct += sum(torch.argmax(model(x.to(device)), dim=1) == y.to(device))
            count += len(y)
    acc = 100. * correct / count
    return acc


def Find_Optimal_Cutoff(target, predicted):
    """
    Adapted from https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]

    return list(roc_t['threshold'])[0]


def auc(model, test_loader, device, c=0, dist="euclidean"):
    output = []
    target = []
    for (x, y) in test_loader:
        x, y = x.to(device), y.to(device)
        feats = model(x)
        output.append(feats)
        target.append(y)

    output = torch.cat(output, dim=0)
    target = torch.cat(target, dim=0)
    if dist == "euclidean":
        scores = torch.mean((output - c) ** 2, dim=1)
    elif dist == "cosine":
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        c = torch.ones(output.shape[1])
        scores = torch.exp(-cos(output, c))
    pred = torch.zeros_like(target)
    threhold = Find_Optimal_Cutoff(target.cpu().detach().numpy(), scores.cpu().detach().numpy())
    threhold = torch.tensor(threhold, requires_grad=False)
    pred[scores > threhold] = 1
    acc = sum(pred == target) * 1.0 / target.shape[0]
    auc = roc_auc_score(target.cpu().detach().numpy(), scores.cpu().detach().numpy())
    return auc, acc.item()
