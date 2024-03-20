from ocpmodels.datasets import TrajectoryLmdbDataset
import numpy as np
import os
import ase.io
from ase.io.extxyz import write_extxyz
from ase import Atoms
from ase.io.db import read_db


dataset = TrajectoryLmdbDataset({"src": "/home/a113/nyx/ocp-main/data/oc20_ni/test_id/data1.lmdb"})

l = len(dataset)
print(l)
k = 1

for i in range(0, l):
    data = dataset[i]
    print(data)
    # print(data.y_relaxed)
    # atom = data.atomic_numbers
    # sid = data.sid
    # cell = data.cell
    # print(data.natoms)
    # print(sid)
    # print(atom)
    # # force = data.force
    # print(data.tags)
    # print(force)
    # y = data.y_relaxed
    # # print(y)
    # # print(data.pos)
    # result = cell.numpy()
    # # print(result)
    # with open("dataset1.txt", "a") as f:
    #     # np.savetxt(f, result, fmt='%d', newline=' ', delimiter=',')
    #     f.write(str(y))
    #     # f.write(k)
    #     f.write("\n")
    # with open("E://ocp-main/data/Ti/dataset.txt", "a") as f:
    #     f.write(str(y))
    #     # f.write(k)
    #     f.write("\n")
    # 保存三维数组
    # with open('a.txt', 'a') as f:
    #     for slice_2d in cell:
    #         f.write(str(k)+':')
    #         f.write(str(y))
    #         f.write("\n")
    #         np.savetxt(f, slice_2d, fmt='%f', delimiter=',')
    #         f.write("\n")
    # k = k+1



