import argparse
import os
import sys
import random
import timeit
import datetime

import numpy as np
import pickle
import scipy.misc
import scipy.io as sio

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data, model_zoo
from torch.autograd import Variable
import torchvision.transforms as transform

from model.deeplabv2 import Res_Deeplab

from utils.loss import CrossEntropy2d
from utils.loss import CrossEntropyLoss2dPixelWiseWeighted
from utils.loss import MSELoss2d
from utils.timer import Timer
from utils import transformmasks
from utils import transformsgpu
from utils.helpers import colorize_mask
import utils.palette as palette
# from utils.random_crop import RandomCrop, RandomCrop_img

from utils.sync_batchnorm import convert_model
from utils.sync_batchnorm import DataParallelWithCallback

from data import get_loader, get_data_path
from data.augmentations import *
from tqdm import tqdm

import PIL
from torchvision import transforms
import json
from torch.utils import tensorboard
from evaluateUDA import evaluate
import os
import os.path as osp
import time

start = timeit.default_timer()
start_writeable = datetime.datetime.now().strftime('%m-%d_%H-%M')

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--gpus", type=int, default=1,
                        help="choose number of gpu devices to use (default: 1)")
    parser.add_argument("-c", "--config", type=str, default='config.json',
                        help='Path to the config file (default: config.json)')
    parser.add_argument("-r", "--resume", type=str, default=None,
                        help='Path to the .pth file to resume from (default: None)')
    parser.add_argument("-n", "--name", type=str, default=None, required=True,
                        help='Name of the run (default: None)')
    parser.add_argument("--save-images", type=str, default=None,
                        help='Include to save images (default: None)')
    return parser.parse_args()



def loss_calc(pred, label):
    label = Variable(label.long()).cuda()
    if len(gpus) > 1:
        criterion = torch.nn.DataParallel(CrossEntropy2d(ignore_label=ignore_label), device_ids=gpus).cuda()  # Ignore label ??
    else:
        criterion = CrossEntropy2d(ignore_label=ignore_label).cuda()  # Ignore label ??

    return criterion(pred, label)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(learning_rate, i_iter, num_iterations, lr_power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def create_ema_model(model):
    #ema_model = getattr(models, config['arch']['type'])(self.train_loader.dataset.num_classes, **config['arch']['args']).to(self.device)
    ema_model = Res_Deeplab(num_classes=num_classes)

    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    #_, availble_gpus = self._get_available_devices(self.config['n_gpu'])
    #ema_model = torch.nn.DataParallel(ema_model, device_ids=availble_gpus)
    if len(gpus)>1:
        #return torch.nn.DataParallel(ema_model, device_ids=gpus)
        if use_sync_batchnorm:
            ema_model = convert_model(ema_model)
            ema_model = DataParallelWithCallback(ema_model, device_ids=gpus)
        else:
            ema_model = torch.nn.DataParallel(ema_model, device_ids=gpus)
    return ema_model

def update_ema_variables(ema_model, model, alpha_teacher, iteration):
    # Use the "true" average until the exponential average is more correct
    alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)
    if len(gpus)>1:
        for ema_param, param in zip(ema_model.module.parameters(), model.module.parameters()):
            #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    else:
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

def strongTransform(parameters, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = transformsgpu.oneMix(mask = parameters["Mix"], data = data, target = target)
    data, target = transformsgpu.colorJitter(colorJitter = parameters["ColorJitter"], img_mean = torch.from_numpy(IMG_MEAN.copy()).cuda(), data = data, target = target)
    data, target = transformsgpu.gaussian_blur(blur = parameters["GaussianBlur"], data = data, target = target)
    data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
    return data, target

def strongTransform_inverse(parameters, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = transformsgpu.oneMix_inverse(mask = parameters["Mix"], data = data, target = target)
    data, target = transformsgpu.colorJitter(colorJitter = parameters["ColorJitter"], img_mean = torch.from_numpy(IMG_MEAN.copy()).cuda(), data = data, target = target)
    data, target = transformsgpu.gaussian_blur(blur = parameters["GaussianBlur"], data = data, target = target)
    data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
    return data, target

def weakTransform(parameters, data=None, target=None):
    data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
    return data, target

def getWeakInverseTransformParameters(parameters):
    return parameters

def getStrongInverseTransformParameters(parameters):
    return parameters

def image_wise_noise(images, noise_scale, noise_max, noise_min):
    return images + torch.clamp(torch.randn_like(images) * noise_scale, noise_min, noise_max)

def sigmoid_ramp_up(iter, max_iter):
    if iter >= max_iter:
        return 1
    else:
        return np.exp(- 5 * (1 - iter / max_iter) ** 2)

def _uncertainty_calc(teacher_net, images, T, max_batch_size, num_classes, noise_scale, noise_min, noise_max):
    # Monte Carlo dropout
    batch_size = images.shape[0]
    all_images_teacher = images.repeat((T, 1, 1, 1))  # [img1, img2, img1, img2, ...]
    all_images_teacher = image_wise_noise(all_images_teacher, noise_scale, noise_min, noise_max)
    all_preds_teacher = None
    # with torch.no_grad():
    #     # Pass everything through with maximum efficiency
    #     j = all_images_teacher.shape[0]
    #     i = 0
    #     while j > i:
    #         cur_batch_size = min(j - i, max_batch_size)
    #         temp = teacher_net(all_images_teacher[i: i + cur_batch_size]).softmax(dim=1)
    #         if all_preds_teacher is None:  # Multi-scale training, this is safer
    #             all_preds_teacher = torch.zeros((T * batch_size, num_classes,
    #                                             temp.shape[2], temp.shape[3])).cuda()
    #         all_preds_teacher[i: i + cur_batch_size] = temp
    #         i += cur_batch_size
    with torch.no_grad():
        all_preds_teacher = F.softmax(teacher_net(all_images_teacher), dim=1)
        # (T x batch-size) x num-classes x H x W => T x batch-size x num-classes x H x W
        all_preds_teacher = all_preds_teacher.reshape(T, batch_size, all_preds_teacher.shape[1],
                                                      all_preds_teacher.shape[2], all_preds_teacher.shape[3])

        averages = all_preds_teacher.mean(dim=0)  # T x batch-size x num-classes x H x W => batch-size x num_classes x H x W
        u = -(averages * averages.log()).sum(dim=1, keepdims=True)  # batch-size x num_classes x H x W => batch-size x H x W
            
    return u


class DeNormalize(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, tensor):
        IMG_MEAN = torch.from_numpy(self.mean.copy())
        IMG_MEAN, _ = torch.broadcast_tensors(IMG_MEAN.unsqueeze(1).unsqueeze(2), tensor)
        tensor = tensor+IMG_MEAN
        tensor = (tensor/255).float()
        tensor = torch.flip(tensor,(0,))
        return tensor

class Learning_Rate_Object(object):
    def __init__(self,learning_rate):
        self.learning_rate = learning_rate

def save_image(image, epoch, id, palette):
    with torch.no_grad():
        if image.shape[0] == 3:
            restore_transform = transforms.Compose([
            DeNormalize(IMG_MEAN),
            transforms.ToPILImage()])


            image = restore_transform(image)
            #image = PIL.Image.fromarray(np.array(image)[:, :, ::-1])  # BGR->RGB
            image.save(os.path.join('./visualiseImages/', str(epoch)+ id + '.png'))
        else:
            mask = image.numpy()
            colorized_mask = colorize_mask(mask, palette)
            colorized_mask.save(os.path.join('./visualiseImages', str(epoch)+ id + '.png'))

def _save_checkpoint(iteration, model, optimizer, config, ema_model, save_best=False, overwrite=True):
    checkpoint = {
        'iteration': iteration,
        'optimizer': optimizer.state_dict(),
        'config': config,
    }
    if len(gpus) > 1:
        checkpoint['model'] = model.module.state_dict()
        if train_unlabeled:
            checkpoint['ema_model'] = ema_model.module.state_dict()
    else:
        checkpoint['model'] = model.state_dict()
        if train_unlabeled:
            checkpoint['ema_model'] = ema_model.state_dict()

    if save_best:
        filename = os.path.join(checkpoint_dir, f'best_model.pth')
        torch.save(checkpoint, filename)
        print("Saving current best model: best_model.pth")
    else:
        filename = os.path.join(checkpoint_dir, f'checkpoint-iter{iteration}.pth')
        print(f'\nSaving a checkpoint: {filename} ...')
        torch.save(checkpoint, filename)
        if overwrite:
            try:
                os.remove(os.path.join(checkpoint_dir, f'checkpoint-iter{iteration - save_checkpoint_every}.pth'))
            except:
                pass

def _resume_checkpoint(resume_path, model, optimizer, ema_model):
    print(f'Loading checkpoint : {resume_path}')
    checkpoint = torch.load(resume_path)

    # Load last run info, the model params, the optimizer and the loggers
    iteration = checkpoint['iteration'] + 1
    print('Starting at iteration: ' + str(iteration))

    if len(gpus) > 1:
        model.module.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])

    optimizer.load_state_dict(checkpoint['optimizer'])

    if train_unlabeled:
        if len(gpus) > 1:
            ema_model.module.load_state_dict(checkpoint['ema_model'])
        else:
            ema_model.load_state_dict(checkpoint['ema_model'])

    return iteration, model, optimizer, ema_model

def main():
    print(config)

    best_mIoU16 = 0
    best_mIoU16_iter = 0 
    best_mIoU13 = 0
    best_mIoU13_iter = 0 
    _t = {'iter time' : Timer()}

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)

    if consistency_loss == 'MSE':
        if len(gpus) > 1:
            unlabeled_loss =  torch.nn.DataParallel(MSELoss2d(), device_ids=gpus).cuda()
        else:
            unlabeled_loss =  MSELoss2d().cuda()
    elif consistency_loss == 'CE':
        if len(gpus) > 1:
            unlabeled_loss = torch.nn.DataParallel(CrossEntropyLoss2dPixelWiseWeighted(ignore_index=ignore_label), device_ids=gpus).cuda()
        else:
            unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted(ignore_index=ignore_label).cuda()

    cudnn.enabled = True

    # create network
    model = Res_Deeplab(num_classes=num_classes)

    # load pretrained parameters
    # saved_state_dict = torch.load(args.restore_from)
    # load pretrained parameters
    if restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(restore_from)
    else:
        saved_state_dict = torch.load(restore_from)

    # Copy loaded parameters to model
    new_params = model.state_dict().copy()
    for name, param in new_params.items():
        if name in saved_state_dict and param.size() == saved_state_dict[name].size():
            new_params[name].copy_(saved_state_dict[name])
    model.load_state_dict(new_params)

    # init ema-model
    if train_unlabeled:
        ema_model = create_ema_model(model)
        ema_model.train()
        ema_model = ema_model.cuda()
    else:
        ema_model = None

    if len(gpus)>1:
        if use_sync_batchnorm:
            model = convert_model(model)
            model = DataParallelWithCallback(model, device_ids=gpus)
        else:
            model = torch.nn.DataParallel(model, device_ids=gpus)
    model.train()
    model.cuda()

    cudnn.benchmark = True
    source_data = 'syn'
    #--------------------------------------------------------------------------------------
    # Data loader for target domain
    if dataset == 'cityscapes':
        data_loader = get_loader('cityscapes')
        data_path = get_data_path('cityscapes')
        if random_crop:
            data_aug = Compose([RandomCrop_city(input_size)])
        else:
            data_aug = None

        #data_aug = Compose([RandomHorizontallyFlip()])
        train_dataset = data_loader(data_path, is_transform=True, augmentations=data_aug, img_size=input_size, img_mean = IMG_MEAN, source=source_data)
        # train_dataset = data_loader(data_path, is_transform=True, augmentations=None, img_size=input_size, img_mean = IMG_MEAN)

    train_dataset_size = len(train_dataset)
    print ('dataset size: ', train_dataset_size)

    if labeled_samples is None:
        trainloader = data.DataLoader(train_dataset,
                        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

        trainloader_remain = data.DataLoader(train_dataset,
                        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        trainloader_remain_iter = iter(trainloader_remain)

    else:
        partial_size = labeled_samples
        print('Training on number of samples:', partial_size)
        np.random.seed(random_seed)
        trainloader_remain = data.DataLoader(train_dataset,
                        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        trainloader_remain_iter = iter(trainloader_remain)

    #--------------------------------------------------------------------------------------
    # Data loader for source domain
    # New loader for Domain transfer
    if True:
        data_loader = get_loader(source_data)
        data_path = get_data_path(source_data)
        if random_crop:
            data_aug = Compose([RandomCrop_gta(input_size)])
        else:
            data_aug = None

        #data_aug = Compose([RandomHorizontallyFlip()])
        train_dataset = data_loader(data_path, list_path = './data/synthia_list/train.txt', augmentations=data_aug, img_size=(1280,760), mean=IMG_MEAN)

    trainloader = data.DataLoader(train_dataset,
                    batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    trainloader_iter = iter(trainloader)
    print('syn size:',len(trainloader))
    #--------------------------------------------------------------------------------------
    # Load new data for domain_transfer

    # optimizer for segmentation network
    learning_rate_object = Learning_Rate_Object(config['training']['learning_rate'])

    if optimizer_type == 'SGD':
        if len(gpus) > 1:
            optimizer = optim.SGD(model.module.optim_parameters(learning_rate_object),
                        lr=learning_rate, momentum=momentum,weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(model.optim_parameters(learning_rate_object),
                        lr=learning_rate, momentum=momentum,weight_decay=weight_decay)
    elif optimizer_type == 'Adam':
        if len(gpus) > 1:
            optimizer = optim.Adam(model.module.optim_parameters(learning_rate_object),
                        lr=learning_rate, momentum=momentum,weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(model.optim_parameters(learning_rate_object),
                        lr=learning_rate, weight_decay=weight_decay)

    optimizer.zero_grad()

    interp = nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)

    start_iteration = 0

    if args.resume:
        start_iteration, model, optimizer, ema_model = _resume_checkpoint(args.resume, model, optimizer, ema_model)

    accumulated_loss_l = []
    accumulated_loss_u = []

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    with open(checkpoint_dir + '/config.json', 'w') as handle:
        json.dump(config, handle, indent=4, sort_keys=True)

    #--------------------------------------------------------------------------------------   
    # start training
    _t['iter time'].tic()
    epochs_since_start = 0
    for i_iter in range(start_iteration, num_iterations):
        model.train()

        loss_u_value = 0
        loss_l_value = 0

        optimizer.zero_grad()

        if lr_schedule:
            adjust_learning_rate(optimizer, i_iter)

        # training loss for labeled data only
        try:
            batch = next(trainloader_iter)
            if batch[0].shape[0] != batch_size:
                batch = next(trainloader_iter)
        except:
            epochs_since_start = epochs_since_start + 1
            print('Epochs since start: ',epochs_since_start)
            trainloader_iter = iter(trainloader)
            batch = next(trainloader_iter)

        # if random_flip:
        #    weak_parameters={"flip":random.randint(0,1)}
        # else:
        weak_parameters={"flip": 0}


        images, labels, _, _, spr = batch 
        images = images.cuda()
        labels = labels.cuda().long()

        # images, labels = weakTransform(weak_parameters, data = images, target = labels)

        pred = interp(model(images))
        L_l = loss_calc(pred, labels) # Cross entropy loss for labeled data
        # L_l = torch.Tensor([0.0]).cuda()

        if train_unlabeled:
            try:
                batch_remain = next(trainloader_remain_iter)
                if batch_remain[0].shape[0] != batch_size:
                    batch_remain = next(trainloader_remain_iter)
            except:
                trainloader_remain_iter = iter(trainloader_remain)
                batch_remain = next(trainloader_remain_iter)
            
            # load the target data and the source spatial prior
            images_remain, _, _, _, _, spr_remain = batch_remain 
            images_remain = images_remain.cuda()
            spr_remain = spr_remain.cuda()

            # flip augmentation for target input
            inputs_u_w, _ = weakTransform(weak_parameters, data = images_remain)
            # inputs_u_w = inputs_u_w.clone()
            # flip augmentation for target prediction
            logits_u_w = interp(ema_model(inputs_u_w))
            logits_u_w, _ = weakTransform(getWeakInverseTransformParameters(weak_parameters), data = logits_u_w.detach())

            spatial_matrix = spr_remain                # torch.Size([1, 19, 512, 512])
            pseudo_label = torch.softmax(logits_u_w.detach(), dim=1)  # torch.Size([2, 19, 512, 512])
            max_probs, targets_u_w = torch.max(pseudo_label, dim=1)

            max_iter = 10000
            # max_iter = num_iterations
            H = (0.75 + 0.25 * sigmoid_ramp_up(i_iter, max_iter)) * math.log(num_classes)
            u_t = _uncertainty_calc(teacher_net=ema_model, images=images_remain, T=8, max_batch_size=16, 
                                    num_classes=num_classes, noise_scale=0.1, noise_min=-0.2, noise_max=0.2)

            u_mask_t = (u_t < H).float()
            u_mask_t = interp(u_mask_t)           # torch.Size([2, 512, 512])
            u_mask_s = torch.ones_like(u_mask_t)  # torch.Size([2, 512, 512])

            # Generate source mask for ClassMix used for warming up
            if i_iter < warm_up_iter:
                for image_i in range(batch_size):
                    # Get all semantic categories that appear in each image
                    classes = torch.unique(labels[image_i])
                    # print('classes', classes) # tensor([  0,   1,   2,   3,   5,   6,   8,   9,  10,  11,  13, 255], device='cuda:0')
                    #classes=classes[classes!=ignore_label]
                    nclasses = classes.shape[0]
                    # print('nclasses', nclasses) # 12
                    #if nclasses > 0:
                    classes = (classes[torch.Tensor(np.random.choice(nclasses, int((nclasses+nclasses%2)/2),replace=False)).long()]).cuda()
                    # Why???
                    if image_i == 0:
                        MixMask0 = transformmasks.generate_class_mask(labels[image_i], classes).unsqueeze(0).cuda()
                    else:
                        MixMask1 = transformmasks.generate_class_mask(labels[image_i], classes).unsqueeze(0).cuda()
            else:
                # Generate source mask for inverse DACS
                if mix_mask == "inverse_simple":
                    for image_i in range(batch_size):
                        # Get all semantic categories that appear in each image
                        classes = torch.unique(targets_u_w[image_i])
                        # print('classes', classes)
                        nclasses = classes.shape[0]

                        classes = (classes[torch.Tensor(np.random.choice(nclasses, int((nclasses+nclasses%2)/2),replace=False)).long()]).cuda()
                        # Why???
                        if image_i == 0:
                            MixMask0 = transformmasks.generate_class_mask(targets_u_w[image_i], classes).unsqueeze(0).cuda()
                        else:
                            MixMask1 = transformmasks.generate_class_mask(targets_u_w[image_i], classes).unsqueeze(0).cuda()

                # Generate source mask for CAMix
                elif mix_mask == "camix":
                    for image_i in range(batch_size):
                        # Get all semantic categories that appear in each image
                        argmax_u_w = torch.mul(spatial_matrix[image_i], pseudo_label[image_i].unsqueeze(0)) # torch.Size([1, 19, 512, 512])
                        argmax_u_w = argmax_u_w.max(1).indices.squeeze(0)  # torch.Size([1, 512, 512]) --> torch.Size([512, 512])
                        classes = torch.unique(argmax_u_w)
                        class_list = classes.cpu().numpy()
                        nclasses = classes.shape[0]
                        categories_index = np.random.choice(nclasses, int((nclasses+nclasses%2)/2),replace=False)
                        categories = class_list[categories_index]
                        categories_new = categories
                        #----------------------------------------------------------------
                        # Group 1 of meta class: object
                        # if contains traffic light, cut the pole
                        if categories_new.__contains__(6) and class_list.__contains__(5):
                            categories_new = np.unique(np.append(categories_new, 5))
                            # print('judge 6, append 5:', categories)
                        # if contains traffic sign, cut the pole
                        if categories_new.__contains__(7) and class_list.__contains__(5):
                            categories_new = np.unique(np.append(categories_new, 5))
                            # print('judge 7, append 5:', categories)
                        # if contains pole, cut the traffic sign and traffic light
                        if categories_new.__contains__(5):
                            if class_list.__contains__(6):
                                categories_new = np.unique(np.append(categories_new, 6))
                                # print('judge 5, append 6:', categories)
                            if class_list.__contains__(7):
                                categories_new = np.unique(np.append(categories_new, 7))
                                # print('judge 5, append 7:', categories)
                        #----------------------------------------------------------------
                        # Group 2 of meta class: human-vehicle
                        # if contains rider, cut the bicycle and motorcycle 
                        if categories_new.__contains__(12):
                            if class_list.__contains__(18):
                                categories_new = np.unique(np.append(categories_new, 18))
                            if class_list.__contains__(17):
                                categories_new = np.unique(np.append(categories_new, 17))
                        # # #----------------------------------------------------------------
                        # # Group 3 of meta class: flat
                        # # if contains road, cut the sidewalk
                        # if categories_new.__contains__(0):
                        #     if class_list.__contains__(1):
                        #         categories_new = np.unique(np.append(categories_new, 1))
                        # # if contains sidewalk, cut the road
                        # if categories_new.__contains__(1):
                        #     if class_list.__contains__(0):
                        #         categories_new = np.unique(np.append(categories_new, 0))                        
                        # # #----------------------------------------------------------------
                        # # Group 4 of meta class: construction
                        # # if contains wall, cut the building 
                        # if categories_new.__contains__(3):
                        #     if class_list.__contains__(2):
                        #         categories_new = np.unique(np.append(categories_new, 2))
                        # # if contains wall, cut the building
                        # if categories_new.__contains__(4):
                        #     if class_list.__contains__(2):
                        #         categories_new = np.unique(np.append(categories_new, 2))
                        # # #----------------------------------------------------------------
                        # # Group 5 of meta class: flat         
                        # # if contains vegetation, cut the terrain
                        # if categories_new.__contains__(8):
                        #     if class_list.__contains__(9):
                        #         categories_new = np.unique(np.append(categories_new, 9))
                        # # if contains terrain, cut the vegetation
                        # if categories_new.__contains__(9):
                        #     if class_list.__contains__(8):
                        #         categories_new = np.unique(np.append(categories_new, 8)) 

                        classes_rt = (torch.Tensor(categories_new).long()).cuda()
                        classes_drop = classes_rt
                        if image_i == 0:
                            MixMask0 = transformmasks.generate_class_mask(argmax_u_w, classes_drop).unsqueeze(0).cuda()
                        else:
                            MixMask1 = transformmasks.generate_class_mask(argmax_u_w, classes_drop).unsqueeze(0).cuda()
                elif mix_mask == None:
                    MixMask = torch.ones((inputs_u_w.shape))
            
            strong_parameters = {"Mix": MixMask0}
            if random_flip:
                strong_parameters["flip"] = random.randint(0, 1)
            else:
                strong_parameters["flip"] = 0
            if color_jitter:
                strong_parameters["ColorJitter"] = random.uniform(0, 1)
            else:
                strong_parameters["ColorJitter"] = 0
            if gaussian_blur:
                strong_parameters["GaussianBlur"] = random.uniform(0, 1)
            else:
                strong_parameters["GaussianBlur"] = 0

            
            # Warm up using ClassMix
            if i_iter < warm_up_iter:
                # Mix up the input images of source and target
                inputs_u_s0, _ = strongTransform(strong_parameters, data = torch.cat((images[0].unsqueeze(0),images_remain[0].unsqueeze(0)))) # torch.Size([1, 3, 512, 512])
                strong_parameters["Mix"] = MixMask1
                inputs_u_s1, _ = strongTransform(strong_parameters, data = torch.cat((images[1].unsqueeze(0),images_remain[1].unsqueeze(0))))
                # Cat inputs_u_s0 & inputs_u_s1
                inputs_u_s = torch.cat((inputs_u_s0,inputs_u_s1))
                logits_u_s = interp(model(inputs_u_s))

                # Mix up the source label and target pseudo_label # torch.Size([2, 512, 512])
                strong_parameters["Mix"] = MixMask0
                _, targets_u0 = strongTransform(strong_parameters, target = torch.cat((labels[0].unsqueeze(0),targets_u_w[0].unsqueeze(0)))) # torch.Size([1, 512, 512])    
                strong_parameters["Mix"] = MixMask1
                _, targets_u1 = strongTransform(strong_parameters, target = torch.cat((labels[1].unsqueeze(0),targets_u_w[1].unsqueeze(0))))
                targets_u = torch.cat((targets_u0,targets_u1)).long()

                if pixel_weight == "threshold_uniform":
                    unlabeled_weight = torch.sum(max_probs.ge(0.968).long() == 1).item() / np.size(np.array(targets_u.cpu()))
                    pixelWiseWeight = unlabeled_weight * torch.ones(max_probs.shape).cuda()
                elif pixel_weight == "threshold":
                    pixelWiseWeight = max_probs.ge(0.968).float().cuda()
                elif pixel_weight == False:
                    pixelWiseWeight = torch.ones(max_probs.shape).cuda()
            
                # Calculte the pixelWiseWeight and cat
                onesWeights = torch.ones((pixelWiseWeight.shape)).cuda()
                strong_parameters["Mix"] = MixMask0
                _, pixelWiseWeight0 = strongTransform(strong_parameters, target = torch.cat((onesWeights[0].unsqueeze(0),pixelWiseWeight[0].unsqueeze(0))))
                strong_parameters["Mix"] = MixMask1
                _, pixelWiseWeight1 = strongTransform(strong_parameters, target = torch.cat((onesWeights[1].unsqueeze(0),pixelWiseWeight[1].unsqueeze(0))))
                pixelWiseWeight = torch.cat((pixelWiseWeight0,pixelWiseWeight1)).cuda()

                # Mix the uncertrainty mask
                strong_parameters["flip"] = 0
                strong_parameters["ColorJitter"] = 0
                strong_parameters["GaussianBlur"] = 0
                strong_parameters["Mix"] = MixMask0
                _, u_mask_mix0 = strongTransform(strong_parameters, target = torch.cat((u_mask_s[0].unsqueeze(0),u_mask_t[0].unsqueeze(0))))
                strong_parameters["Mix"] = MixMask1
                _, u_mask_mix1 = strongTransform(strong_parameters, target = torch.cat((u_mask_s[1].unsqueeze(0),u_mask_t[1].unsqueeze(0))))
                u_mask_mix = torch.cat((u_mask_mix0, u_mask_mix1)).cuda()
                total_instances = u_mask_mix.sum() 
            
            # Normal Training with CAMix
            else:
                # Mix up the input images of source and target
                inputs_u_s0, _ = strongTransform_inverse(strong_parameters, data = torch.cat((images[0].unsqueeze(0),images_remain[0].unsqueeze(0))))
                # print('inputs_u_s0.shape', inputs_u_s0.shape)     # torch.Size([1, 3, 512, 512])
                strong_parameters["Mix"] = MixMask1
                inputs_u_s1, _ = strongTransform_inverse(strong_parameters, data = torch.cat((images[1].unsqueeze(0),images_remain[1].unsqueeze(0))))
                # Cat inputs_u_s0 & inputs_u_s1
                inputs_u_s = torch.cat((inputs_u_s0,inputs_u_s1))
                logits_u_s = interp(model(inputs_u_s))

                # Mix up the source label and target pseudo_label    # torch.Size([2, 512, 512])
                strong_parameters["Mix"] = MixMask0
                _, targets_u0 = strongTransform_inverse(strong_parameters, target = torch.cat((labels[0].unsqueeze(0),targets_u_w[0].unsqueeze(0))))    # torch.Size([1, 512, 512])    
                strong_parameters["Mix"] = MixMask1
                _, targets_u1 = strongTransform_inverse(strong_parameters, target = torch.cat((labels[1].unsqueeze(0),targets_u_w[1].unsqueeze(0))))
                targets_u = torch.cat((targets_u0,targets_u1)).long()

                if pixel_weight == "threshold_uniform":
                    unlabeled_weight = torch.sum(max_probs.ge(0.968).long() == 1).item() / np.size(np.array(targets_u.cpu()))
                    pixelWiseWeight = unlabeled_weight * torch.ones(max_probs.shape).cuda()
                elif pixel_weight == "threshold":
                    pixelWiseWeight = max_probs.ge(0.968).float().cuda()
                elif pixel_weight == False:
                    pixelWiseWeight = torch.ones(max_probs.shape).cuda()
            
                # Calculte the pixelWiseWeight and cat
                onesWeights = torch.ones((pixelWiseWeight.shape)).cuda()
                strong_parameters["Mix"] = MixMask0
                _, pixelWiseWeight0 = strongTransform_inverse(strong_parameters, target = torch.cat((onesWeights[0].unsqueeze(0),pixelWiseWeight[0].unsqueeze(0))))
                strong_parameters["Mix"] = MixMask1
                _, pixelWiseWeight1 = strongTransform_inverse(strong_parameters, target = torch.cat((onesWeights[1].unsqueeze(0),pixelWiseWeight[1].unsqueeze(0))))
                pixelWiseWeight = torch.cat((pixelWiseWeight0,pixelWiseWeight1)).cuda()

                # Mix the uncertrainty mask
                strong_parameters["flip"] = 0
                strong_parameters["ColorJitter"] = 0
                strong_parameters["GaussianBlur"] = 0
                strong_parameters["Mix"] = MixMask0
                _, u_mask_mix0 = strongTransform_inverse(strong_parameters, target = torch.cat((u_mask_s[0].unsqueeze(0),u_mask_t[0].unsqueeze(0))))
                strong_parameters["Mix"] = MixMask1
                _, u_mask_mix1 = strongTransform_inverse(strong_parameters, target = torch.cat((u_mask_s[1].unsqueeze(0),u_mask_t[1].unsqueeze(0))))
                u_mask_mix = torch.cat((u_mask_mix0, u_mask_mix1)).cuda()
                total_instances = u_mask_mix.sum() 

            if consistency_loss == 'MSE':
                unlabeled_weight = torch.sum(max_probs.ge(0.968).long() == 1).item() / np.size(np.array(targets_u.cpu()))
                #pseudo_label = torch.cat((pseudo_label[1].unsqueeze(0),pseudo_label[0].unsqueeze(0)))
                L_u = consistency_weight * unlabeled_weight * unlabeled_loss(logits_u_s, pseudo_label)
            elif consistency_loss == 'CE':
                L_u = consistency_weight * unlabeled_loss(logits_u_s, targets_u, pixelWiseWeight) * u_mask_mix
                if total_instances == 0:
                    L_u = L_u.sum()
                else:
                    L_u = L_u.sum() / total_instances

            loss = L_l + L_u

        else:
            loss = L_l

        if len(gpus) > 1:
            #print('before mean = ',loss)
            loss = loss.mean()
            #print('after mean = ',loss)
            loss_l_value += L_l.mean().item()
            if train_unlabeled:
                loss_u_value += L_u.mean().item()
        else:
            loss_l_value += L_l.item()
            if train_unlabeled:
                loss_u_value += L_u.item()

        loss.backward()
        optimizer.step()

        # update Mean teacher network
        if ema_model is not None:
            alpha_teacher = 0.99
            ema_model = update_ema_variables(ema_model = ema_model, model = model, alpha_teacher=alpha_teacher, iteration=i_iter)
        
        print(checkpoint_dir)
        _t['iter time'].toc(average=False)
        print('iter = {0:6d}/{1:6d}, loss_l = {2:.3f}, loss_u = {3:.3f}, iter time = {4:.2f} s'.format(
            i_iter, num_iterations, loss_l_value, loss_u_value, _t['iter time'].diff))

        if i_iter % save_checkpoint_every == 0 and i_iter!=0:
            if epochs_since_start * len(trainloader) < save_checkpoint_every:
                _save_checkpoint(i_iter, model, optimizer, config, ema_model, overwrite=False)
            else:
                _save_checkpoint(i_iter, model, optimizer, config, ema_model)
                # if i_iter > 200000:
                #     _save_checkpoint(i_iter, model, optimizer, config, ema_model, overwrite=False)
                # else:
                #     _save_checkpoint(i_iter, model, optimizer, config, ema_model)

        if config['utils']['tensorboard']:
            if 'tensorboard_writer' not in locals():
                tensorboard_writer = tensorboard.SummaryWriter(log_dir, flush_secs=30)

            accumulated_loss_l.append(loss_l_value)
            if train_unlabeled:
                accumulated_loss_u.append(loss_u_value)
            if i_iter % log_per_iter == 0 and i_iter != 0:

                tensorboard_writer.add_scalar('Training/Supervised loss', np.mean(accumulated_loss_l), i_iter)
                accumulated_loss_l = []

                if train_unlabeled:
                    tensorboard_writer.add_scalar('Training/Unsupervised loss', np.mean(accumulated_loss_u), i_iter)
                    accumulated_loss_u = []


        if i_iter % val_per_iter == 0 and i_iter != 0:
            model.eval()
            if dataset == 'cityscapes':
                _, mIoU16, mIoU13 = evaluate(model, dataset, source_data=source_data, i_iter=i_iter, ignore_label=255, input_size=(512,1024), save_dir=checkpoint_dir)
                print('Cur mIoU16', mIoU16)
                print('Cur mIoU13', mIoU13)
                print('Cur mIoU16 iter', i_iter)
                print('Cur mIoU13 iter', i_iter)

                f_mIoU16 = open(osp.join(checkpoint_dir, 'mIoU16.txt'), 'a')
                # f_mIoU16.write('{0:.4f}\n'.format(mIoU16))
                f_mIoU16.write('{0:d} steps: {1:.4f}\n'.format(i_iter, mIoU16))
                f_mIoU16.close()            
                
                f_mIoU13 = open(osp.join(checkpoint_dir, 'mIoU13.txt'), 'a')
                # f_mIoU13.write('{0:.4f}\n'.format(mIoU13))
                f_mIoU13.write('{0:d} steps: {1:.4f}\n'.format(i_iter, mIoU13))
                f_mIoU13.close()           

            model.train()

            if mIoU16 > best_mIoU16 and save_best_model:
                best_mIoU16 = mIoU16
                best_mIoU16_iter = i_iter
                print('Best mIoU16', best_mIoU16)
                print('Best mIoU16 iter', best_mIoU16_iter)
                _save_checkpoint(i_iter, model, optimizer, config, ema_model, save_best=True)

            if mIoU13 > best_mIoU13 and save_best_model:
                best_mIoU13 = mIoU13
                best_mIoU13_iter = i_iter
                print('Best mIoU13', best_mIoU13)
                print('Best mIoU13 iter', best_mIoU13_iter)

            if config['utils']['tensorboard']:
                tensorboard_writer.add_scalar('Validation/mIoU16', mIoU16, i_iter)
                tensorboard_writer.add_scalar('Validation/mIoU13', mIoU13, i_iter)

        if save_unlabeled_images and train_unlabeled and i_iter % save_checkpoint_every == 0:
            # Saves two mixed images and the corresponding prediction
            save_image(inputs_u_s[0].cpu(),i_iter,'input1',palette.CityScpates_palette)
            save_image(inputs_u_s[1].cpu(),i_iter,'input2',palette.CityScpates_palette)
            _, pred_u_s = torch.max(logits_u_s, dim=1)
            save_image(pred_u_s[0].cpu(),i_iter,'pred1',palette.CityScpates_palette)
            save_image(pred_u_s[1].cpu(),i_iter,'pred2',palette.CityScpates_palette)

    _save_checkpoint(num_iterations, model, optimizer, config, ema_model)

    model.eval()
    if dataset == 'cityscapes':
        _, mIoU16, mIoU13 = evaluate(model, dataset, source_data=source_data, i_iter=num_iterations, ignore_label=255, input_size=(512,1024), save_dir=checkpoint_dir)
        print('Cur mIoU16', mIoU16)
        print('Cur mIoU13', mIoU13)
        print('Cur mIoU16 iter', i_iter)
        print('Cur mIoU13 iter', i_iter)

        f_mIoU16 = open(osp.join(checkpoint_dir, 'mIoU16.txt'), 'a')
        # f_mIoU16.write('{0:.4f}\n'.format(mIoU16))
        f_mIoU16.write('{0:d} steps: {1:.4f}\n'.format(i_iter, mIoU16))
        f_mIoU16.close()              
        
        f_mIoU13 = open(osp.join(checkpoint_dir, 'mIoU13.txt'), 'a')
        # f_mIoU13.write('{0:.4f}\n'.format(mIoU13))
        f_mIoU13.write('{0:d} steps: {1:.4f}\n'.format(i_iter, mIoU13))
        f_mIoU13.close()  

    if mIoU16 > best_mIoU16 and save_best_model:
        best_mIoU16 = mIoU16
        best_mIoU16_iter = i_iter
        print('Best mIoU16', best_mIoU16)
        print('Best mIoU16 iter', best_mIoU16_iter)
        _save_checkpoint(i_iter, model, optimizer, config, ema_model, save_best=True)

    if mIoU13 > best_mIoU13 and save_best_model:
        best_mIoU13 = mIoU13
        best_mIoU13_iter = i_iter
        print('Best mIoU13', best_mIoU13)
        print('Best mIoU13 iter', best_mIoU13_iter)

    if config['utils']['tensorboard']:
        tensorboard_writer.add_scalar('Validation/mIoU16', mIoU16, i_iter)
        tensorboard_writer.add_scalar('Validation/mIoU13', mIoU13, i_iter)

    end = timeit.default_timer()
    print('Total time: ' + str(end-start) + 'seconds')

if __name__ == '__main__':

    print('---------------------------------Starting---------------------------------')

    args = get_arguments()

    if False:#args.resume:
        config = torch.load(args.resume)['config']
    else:
        config = json.load(open(args.config))

    model = config['model']
    dataset = config['dataset']


    if config['pretrained'] == 'coco':
        # restore_from = 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth'
        restore_from = './model/resnet101COCO-41f33a49.pth'

    num_classes=19
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

    batch_size = config['training']['batch_size']
    num_iterations = config['training']['num_iterations']

    input_size_string = config['training']['data']['input_size']
    h, w = map(int, input_size_string.split(','))
    input_size = (h, w)

    ignore_label = config['ignore_label'] 

    learning_rate = config['training']['learning_rate']

    optimizer_type = config['training']['optimizer']
    lr_schedule = config['training']['lr_schedule']
    lr_power = config['training']['lr_schedule_power']
    weight_decay = config['training']['weight_decay']
    momentum = config['training']['momentum']
    num_workers = config['training']['num_workers']
    use_sync_batchnorm = config['training']['use_sync_batchnorm']
    random_seed = config['seed']

    labeled_samples = config['training']['data']['labeled_samples']

    #unlabeled CONFIGURATIONS
    train_unlabeled = config['training']['unlabeled']['train_unlabeled']
    mix_mask = config['training']['unlabeled']['mix_mask']
    pixel_weight = config['training']['unlabeled']['pixel_weight']
    consistency_loss = config['training']['unlabeled']['consistency_loss']
    consistency_weight = config['training']['unlabeled']['consistency_weight']
    random_flip = config['training']['unlabeled']['flip']
    color_jitter = config['training']['unlabeled']['color_jitter']
    gaussian_blur = config['training']['unlabeled']['blur']

    random_scale = config['training']['data']['scale']
    random_crop = config['training']['data']['crop']

    save_checkpoint_every = config['utils']['save_checkpoint_every']
    if args.resume:
        checkpoint_dir = os.path.join(*args.resume.split('/')[:-1]) + '_resume-' + start_writeable
    else:
        checkpoint_dir = os.path.join(config['utils']['checkpoint_dir'], start_writeable)
    log_dir = checkpoint_dir

    warm_up_iter = config['utils']['warm_up_iter']
    val_per_iter = config['utils']['val_per_iter']
    use_tensorboard = config['utils']['tensorboard']
    log_per_iter = config['utils']['log_per_iter']

    save_best_model = config['utils']['save_best_model']
    if args.save_images:
        print('Saving unlabeled images')
        save_unlabeled_images = True
    else:
        save_unlabeled_images = False

    gpus = (0,1,2,3)[:args.gpus]

    main()
