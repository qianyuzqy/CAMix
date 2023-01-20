import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image

#gqq
import scipy.io as sio
import copy


class SynthiaDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, augmentations = None, img_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=250):
        self.root = root
        self.list_path = list_path
        self.img_size = img_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.augmentations = augmentations
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip()[-11:] for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

        self.id_to_trainid = {3: 0, 4: 1, 2: 2, 21: 3, 5: 4, 7: 5,
                              15: 6, 9: 7, 6: 8, 16: 9, 1: 10, 10: 11, 17: 12,
                              8: 13, 18: 14, 19: 15, 20: 16, 12: 17, 11: 18}

        #for name in self.img_ids:
        #    img_file = osp.join(self.root, "RGB/%s" % name)
        #    label_file = osp.join(self.root, "GT/LABELS/%s" % name)
        #    self.files.append({
        #        "img": img_file,
        #        "label": label_file,
        #        "name": name
        #    })

        #gqq
        self.spr = self.get_spatial_matrix().numpy().transpose(1, 2, 0)#(1024, 2048, 19)
        #print(type(self.spr), self.spr.shape, self.spr.dtype) #<class 'numpy.ndarray'> (1024, 2048, 19) float32


    #gqq
    def get_spatial_matrix(self, spr_path= "../model/prior_array.mat" ):
        if not os.path.exists(spr_path):
            raise FileExistsError("please put the spatial prior in ..model/")
        sprior = sio.loadmat(spr_path)
        sprior = sprior["prior_array"]
        tensor_sprior = torch.tensor(sprior, dtype=torch.float64).float()#.cuda()
        return tensor_sprior

    def __len__(self):
        return len(self.img_ids)


    def __getitem__(self, index):
        name = self.img_ids[index]
        image = Image.open(osp.join(self.root, "RGB/%s" % name)).convert('RGB')
        label = Image.open(osp.join(self.root, "synthia_mapped_to_cityscapes/%s" % name))

        # resize
        image = image.resize(self.img_size, Image.BICUBIC)
        label = label.resize(self.img_size, Image.NEAREST)

        image = np.asarray(image, np.uint8)
        label = np.asarray(label, np.uint8)

        if self.augmentations is not None:
            image, label, position = self.augmentations(image, label) #gqq

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        #gqq--------------
        spr_crop = self.spr[position[1]:position[3], position[0]:position[2],:]
        spr_crop = self.transform_spr(spr_crop)

        return image.copy(), label_copy.copy(), np.array(size), name, spr_crop #gqq


    #gqq
    def transform_spr(self, spr):
        spr = spr.transpose(2, 0, 1)
        spr = torch.from_numpy(spr).float()
        return spr

if __name__ == '__main__':
    dst = GTA5DataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
