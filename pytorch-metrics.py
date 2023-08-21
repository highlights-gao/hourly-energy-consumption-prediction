# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 11:12:07 2023

@author: gaoji
"""
from torchmetrics import R2Score
import torch
import numpy as np
from sklearn.metrics import r2_score


target = torch.tensor([1,2,3])
pred = torch.tensor([1.1,1.2,3.5])
r2score = R2Score()
s1 = r2score(pred, target).values



targets = torch.tensor([[0.5, 1], [-1, 1], [7, -6]])
preds = torch.tensor([[0, 2], [-1, 2], [8, -5]])
r2score = R2Score(num_outputs=2, multioutput='raw_values')
s2 = r2score(preds, targets)



target3 = np.array(([4,5,6],[7,8,9]))
pred3 = np.array(([7,5,3],[7,8,9]))

s3 = r2_score(target3,pred3)

