import os
import torch
import numpy as np
import scipy.misc as m
import torch.nn as nn

#gqq
import scipy.io as sio
import copy

from torch.utils import data

from data.city_utils import recursive_glob
from data.augmentations import *

class cityscapesLoader(data.Dataset):
    """cityscapesLoader

    https://www.cityscapes-dataset.com

    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/

    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """

    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))

    mean_rgb = {"cityscapes": [73.15835921, 82.90891754, 72.39239876],}

    def __init__(
        self,
        root,
        source,
        split="train",
        is_transform=False,
        img_size=(512, 1024),
        img_norm=False,
        augmentations=None,
        version="cityscapes",
        return_id=False,
        img_mean = np.array([73.15835921, 82.90891754, 72.39239876]),
    ):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.root = root
        self.split = split
        self.source = source
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 19
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )
        self.mean = img_mean
        self.files = {}

        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.annotations_base = os.path.join(
            self.root, "gtFine", self.split
        )

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33,]
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if not self.files[split]:
            raise Exception(
                "No files for split=[%s] found in %s" % (split, self.images_base)
            )

        print("Found %d %s images" % (len(self.files[split]), split))

        self.return_id = return_id

        if self.source == "gtav":
            print('==============source = gtav')
            self.spr = self.get_spatial_matrix_gtav().numpy().transpose(1, 2, 0) #(1024, 2048, 19)
        elif self.source == "syn":
            print('==============source = syn')
            self.spr = self.get_spatial_matrix_syn()
            interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)
            self.spr = self.spr.unsqueeze(0)
            self.spr = interp(self.spr).squeeze()
            self.spr = self.spr.numpy().transpose(1, 2, 0)  #(760, 1280, 19)
            # print(self.spr.shape)

        # #gqq
        # self.spr = self.get_spatial_matrix().numpy().transpose(1, 2, 0)#(1024, 2048, 19)
        # #print(type(self.spr), self.spr.shape, self.spr.dtype) #<class 'numpy.ndarray'> (1024, 2048, 19) float32

    #gqq
    def get_spatial_matrix_gtav(self, spr_path= "../model/prior_array.mat" ):
        if not os.path.exists(spr_path):
            raise FileExistsError("please put the spatial prior in ..model/")
        sprior = sio.loadmat(spr_path)
        sprior = sprior["prior_array"]
        # foreground_map = [5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18]
        # background_map = [0, 1, 2, 3, 4, 8, 9, 10]
        # sprior = sprior[foreground_map]
        # sprior = sprior[background_map]
        tensor_sprior = torch.tensor(sprior, dtype=torch.float64).float()#.cuda()
        return tensor_sprior

    def get_spatial_matrix_syn(self, spr_path='../model/spatial_prior_19_syn/probarray_cls{0:d}.mat'.format(0)):
        # if not os.path.exists(path):
        #     raise FileExistsError("please put the spatial prior in ..model/")
        # path = '../model/spatial_prior_19_syn/probarray_cls{0:d}.mat'.format(0)
        sprior = sio.loadmat(spr_path)
        sprior = sprior["array"]
        tensor_sprior_cat = torch.tensor(sprior, dtype=torch.float64).float().unsqueeze(0)
        # print('tensor_sprior_cat.shape', tensor_sprior_cat.shape)
        for i in range(1, 19):
            path = '../model/spatial_prior_19_syn/probarray_cls{0:d}.mat'.format(i)
            sprior = sio.loadmat(path)
            sprior = sprior["array"]
            tensor_sprior_cur = torch.tensor(sprior, dtype=torch.float64).float().unsqueeze(0)     
            tensor_sprior_cat = torch.cat((tensor_sprior_cat, tensor_sprior_cur))
        return tensor_sprior_cat

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2], # temporary for cross validation
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )

        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)#(1024, 2048, 3)

        lbl = m.imread(lbl_path)
        lbl = np.array(lbl, dtype=np.uint8)
        lbl = self.encode_segmap(lbl)#(1024, 2048)
        #print('aaa', img.shape, lbl.shape, self.spr.shape)

        if self.augmentations is not None:
            img, lbl, position = self.augmentations(img, lbl)

        # #gqq--------------
        # #position: x1, y1, x2, y2
        # spr_crop = self.spr[position[1]:position[3], position[0]:position[2],:]
        # print(position, spr_crop.shape)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)
            if self.split=='val':
                img_name = img_path.split('/')[-1]
                if self.return_id:
                    return img, lbl, img_name, img_name, index 
                return img, lbl, img_path, lbl_path, img_name               
            elif self.split=='train':
                #print('self.spr.shape', self.spr.shape, position)
                spr_crop = self.spr[position[1]:position[3], position[0]:position[2],:]
                spr_crop = self.transform_spr(spr_crop)

                #print('bbb', img.shape, lbl.shape, spr_crop.shape)
                #img torch.Size([3, 512, 512]) lbl torch.Size([512, 512]) torch.Size([19, 512, 512])
                #end--------------
           
                img_name = img_path.split('/')[-1]
                # print('img.shape', img.shape)
                # print('lbl.shape', lbl.shape)
                # print('img_name', img_name)
                # print('spr_crop.shape', spr_crop.shape)
                if self.return_id:
                    return img, lbl, img_name, img_name, index, spr_crop #gqq
                return img, lbl, img_path, lbl_path, img_name, spr_crop #gqq

    #gqq
    def transform_spr(self, spr):
        spr = spr.transpose(2, 0, 1)
        spr = torch.from_numpy(spr).float()
        return spr


    def transform(self, img, lbl):
        """transform

        :param img:
        :param lbl:
        """
        img = m.imresize(
            img, (self.img_size[0], self.img_size[1])
        )  # uint8 with RGB mode
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        #print(type(img))
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        lbl = lbl.astype(int)
        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

'''
if __name__ == "__main__":
    import torchvision
    import matplotlib.pyplot as plt

    augmentations = Compose([Scale(2048), RandomRotate(10), RandomHorizontallyFlip()])

    local_path = "./data/city_dataset/"
    dst = cityscapesLoader(local_path, is_transform=True, augmentations=augmentations)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
        plt.show()
        a = raw_input()
        if a == "ex":
            break
        else:
            plt.close()
'''
