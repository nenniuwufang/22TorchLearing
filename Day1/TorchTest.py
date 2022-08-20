# -*- coding:utf-8 -*-
import torch

x = torch.zeros(2, 1, 2, 1, 2)
print(x.size())

y = torch.squeeze(x)
print(y.size())

y = torch.squeeze(x, 0)
print(y.size())

y = torch.squeeze(x, 1)
print(y.size())
