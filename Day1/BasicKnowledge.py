# -*- coding:utf-8 -*-
import torch

z = torch.Tensor(4, 5)
print(z)

y = torch.rand(4, 5)
print(z + y)

print(torch.add(z, y))

b = z.numpy()
print(b)




