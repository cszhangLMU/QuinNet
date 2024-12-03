import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from collections import OrderedDict
from glob import glob
from torch.utils.data import random_split

import torch
import torch.backends.cudnn as cudnn
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from albumentations import RandomRotate90,Resize, Flip, Compose
import albumentations as A
from albumentations.pytorch import ToTensorV2
import archsSEMlp
from archs import Resnet34_Unet, MTUNet,Guide_UNext,UNetplus, UNet, AttUNet,UNext,PAttUNet,FCT
import losses
from dataset import Dataset, Data_loaderVE
from metrics import iou_score,SegmentationMetric, NT_Xent
from utils import AverageMeter, str2bool
import torchvision.transforms.functional as TF
#import torch.backends.cudnn as cudnn

ARCH_NAMES = archsSEMlp.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='QuinNet')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int, help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=512, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=512, type=int,
                        help='image height')
    
    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    
    # dataset
    parser.add_argument('--dataset', default='ISIC2018')
    parser.add_argument('--img_ext', default='.jpg',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )

    parser.add_argument('--num_workers', default=0, type=int)

    config = parser.parse_args()

    return config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# args = parser.parse_args()
def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}
    model.train()
    pbar = tqdm(total=len(train_loader))
    infoLoss = NT_Xent(temperature=0.5)
    #cudnn.benchmark = True
    for input, target, _ in train_loader:
        input = input.float().to(device)
        target = target.float().to(device)
        if config['dataset'] == 'Covid_Infection':
            target = target.unsqueeze(dim=1)
        # compute output
        if config['deep_supervision']: # False

            outputs = model(input)
            print('===================')
            print(input.shape)
            print(target.shape)
            print(outputs.shape)
            loss = 0
            for output in outputs:
                #print(output.shape)
                loss += criterion(output, target)
            loss /= len(outputs)
            iou,dice = iou_score(outputs[-1], target)
            loss = 0.5*loss + dice
        else:
            if config['arch'] == 'QuinNet':
                mspOutput1,mspOutput2,mspOutput3,mspOutput4,output = model(input)
                #print(output.shape)
                _,_,H,W = target.shape
                #loss = criterion(output, target)+criterion(mspOutput1,TF.resize(target,size=[int(H/2),int(W/2)]))+criterion(mspOutput2,TF.resize(target,size=[int(H/4),int(W/4)]))+criterion(mspOutput3,TF.resize(target,size=[int(H/8),int(W/8)]))+criterion(mspOutput4,TF.resize(target,size=[int(H/16),int(W/16)]))
                loss = criterion(output, target)+criterion(mspOutput1,target)+criterion(mspOutput2,target)+criterion(mspOutput3,target)+criterion(mspOutput4,target)
                iou, dice = iou_score(output, target)
                loss = 0.5*loss + dice
            elif config['arch'] == 'SkipConnect_SACA_UNet_crop_resize':
                targetArea = []
                for i in range(target.shape[0]):
                    temp = []
                    axis = torch.nonzero(target[i])
                    maxX_Y,_ = torch.max(axis,dim=0)
                    minX_Y,_ = torch.min(axis,dim=0)
                    temp.append(int(minX_Y[1]))
                    temp.append(int(minX_Y[2]))
                    temp.append(int(maxX_Y[1]))
                    temp.append(int(maxX_Y[2]))
                    targetArea.append(temp)
                #####
                output = model(input,targetArea)
                loss = criterion(output, target)
                iou, dice = iou_score(output, target)
                loss = 0.5*loss + dice
            else:
                if config['arch'] == 'nnFormer':
                    input = input.unsqueeze(2)  # 将 (N, H, W) -> (N, C=1, H, W)
                    target = target.unsqueeze(2)  # 将 (N, H, W) -> (N, C=1, H, W)
                output = model(input)
                #print(output.shape)
                #print(target.shape)
                loss = criterion(output, target)
                iou, dice = iou_score(output, target)
                loss = 0.5*loss + dice

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter(),
                  'Acc': AverageMeter(),
                  'mAcc': AverageMeter(),
                  'mIoU': AverageMeter(),
                  'recall': AverageMeter(),
                  'spec': AverageMeter()}
                   
    metric = SegmentationMetric(config['num_classes'])
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            
            input = input.float().to(device)
            target = target.float().to(device)
            if config['dataset'] == 'Covid_Infection':
                target = target.unsqueeze(dim=1)
            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou,dice = iou_score(outputs[-1], target)
            else:
                if config['arch'] == 'QuinNet':
                    mspOutput1,mspOutput2,mspOutput3,mspOutput4,output = model(input)
                    _,_,H,W = target.shape
                    #loss = criterion(output, target)+criterion(mspOutput1,TF.resize(target,size=[int(H/2),int(W/2)]))+criterion(mspOutput2,TF.resize(target,size=[int(H/4),int(W/4)]))+criterion(mspOutput3,TF.resize(target,size=[int(H/8),int(W/8)]))+criterion(mspOutput4,TF.resize(target,size=[int(H/16),int(W/16)]))
                    loss = criterion(output, target)+criterion(mspOutput1,target)+criterion(mspOutput2,target)+criterion(mspOutput3,target)+criterion(mspOutput4,target)
                    iou, dice = iou_score(output, target)
                    loss = 0.5*loss + dice
                elif config['arch'] == 'SkipConnect_SACA_UNet_crop_resize':
                    output = model(input)
                    loss = criterion(output, target)
                    iou, dice = iou_score(output, target)
                    loss = 0.5*loss + dice
                else:
                    output = model(input)
                    loss = criterion(output, target)
                    iou,dice = iou_score(output, target)
                    loss = 0.5*loss + dice
                
                metric.addBatch(output, target)
                PA = metric.pixelAccuracy()
                mPA = metric.meanPixelAccuracy()
                mIoU = metric.meanIntersectionOverUnion()
                recall = metric.computeRecall()
                spec = metric.computeSpec()
                #metric.reset()
            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['Acc'].update(PA, input.size(0))
            avg_meters['mAcc'].update(mPA, input.size(0))
            avg_meters['mIoU'].update(mIoU, input.size(0))
            avg_meters['recall'].update(recall, input.size(0))
            avg_meters['spec'].update(spec, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
                ('Acc', avg_meters['Acc'].avg),
                ('mAcc', avg_meters['mAcc'].avg),
                ('mIoU', avg_meters['mIoU'].avg),
                ('recall', avg_meters['recall'].avg),
                ('spec', avg_meters['spec'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg),
                        ('Acc', avg_meters['Acc'].avg),
                        ('mAcc', avg_meters['mAcc'].avg),
                        ('mIoU', avg_meters['mIoU'].avg),
                        ('recall', avg_meters['recall'].avg),
                        ('spec', avg_meters['spec'].avg)
                        ])
                        


def main():
    config = vars(parse_args())
    writer = SummaryWriter(log_dir='runs/{}_{}_log'.format(config['arch'],config['name']))
    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    #print('-' * 20)
    #for key in config:
        #print('%s: %s' % (key, config[key]))
    #print('-' * 20)

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)
    
    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().to(device)
    else:
        criterion = losses.__dict__[config['loss']]().to(device)

    cudnn.benchmark = True

    # Data loading code
    ##img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    ##img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    train_dir = ''
    val_dir = ''
    train_mask_dir = ''
    val_mask_dir = ''
    train_img_ = []
    val_img_ = []
    if config['dataset'] == 'ISIC2018':
        # Train data loading code
        train_img_ids_ = glob(os.path.join('./dataset/ISIC2018/Training_Input', '*' + '.jpg'))
        #train_img_ids_ = glob(os.path.join('../dataset/ISIC2018/Training_Input', '*' + '.jpg'))
        train_img_ids_ = [os.path.splitext(os.path.basename(p))[0] for p in train_img_ids_]
        train_img_ = train_img_ids_
        # Valition data loading code
        val_img_ids_ = glob(os.path.join('./dataset/ISIC2018/Validation_Input', '*' + '.jpg'))
        #val_img_ids_ = glob(os.path.join('../dataset/ISIC2018/Validation_Input', '*' + '.jpg'))
        val_img_ids_ = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids_]
        val_img_ = val_img_ids_
        train_dir = './dataset/ISIC2018/Training_Input'
        train_mask_dir = os.path.join('./dataset/ISIC2018', 'Training_GroundTruth')
        val_dir = './dataset/ISIC2018/Validation_Input'
        val_mask_dir = os.path.join('./dataset/ISIC2018', 'Validation_GroundTruth')
        config['img_ext'] = '.jpg'
    elif config['dataset'] == 'ISIC2018NEW2':
        # Train data loading code
        train_img_ids_ = glob(os.path.join('../dataset/ISIC2018NEW2/Train_Folder_images', '*' + '.jpg'))
        train_img_ids_ = [os.path.splitext(os.path.basename(p))[0] for p in train_img_ids_]
        train_img_ = train_img_ids_
        # Valition data loading code
        val_img_ids_ = glob(os.path.join('../dataset/ISIC2018NEW2/Test_Folder_images', '*' + '.jpg'))
        val_img_ids_ = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids_]
        val_img_ = val_img_ids_
        train_dir = '../dataset/ISIC2018NEW2/Train_Folder_images'
        train_mask_dir = os.path.join('../dataset/ISIC2018NEW2', 'Train_Folder_masks')
        val_dir = '../dataset/ISIC2018NEW2/Test_Folder_images'
        val_mask_dir = os.path.join('../dataset/ISIC2018NEW2', 'Test_Folder_masks')
        config['img_ext'] = '.jpg'
    elif config['dataset'] == 'BUSI':
        train_img_ids_ = glob(os.path.join('./dataset/BUSI/train', '*' + '.png'))
        train_img_ids_ = [os.path.splitext(os.path.basename(p))[0] for p in train_img_ids_]
        for p in train_img_ids_:
            m = p.split('_')
            if m[-1] != 'mask':
                train_img_.append(m[0])
        val_img_ids_ = glob(os.path.join('./dataset/BUSI/val', '*' + '.png'))
        val_img_ids_ = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids_]
        
        for p in val_img_ids_:
            m = p.split('_')
            if m[-1] != 'mask':
                val_img_.append(m[0])
        train_dir = './dataset/BUSI/train'
        train_mask_dir = train_dir
        val_dir = './dataset/BUSI/val'
        val_mask_dir = val_dir
        config['img_ext'] = '.png'
    elif config['dataset'] == 'cvc-clinicdb':
        train_dir = './dataset/CVC-ClinicDB/PNG/Original/'
        train_mask_dir = './dataset/CVC-ClinicDB/PNG/Ground_Truth/'
        #train_dir = './cvc-clinicdb/PNG/Original'
        #train_mask_dir = './cvc-clinicdb/PNG/Ground_Truth'
        train_img_ids_ = glob(os.path.join(train_dir, '*' + '.png'))
        train_img_ids_ = [os.path.splitext(os.path.basename(p))[0] for p in train_img_ids_]
    elif config['dataset'] == 'Kvasir-SEG':
        train_dir = '../dataset/Kvasir-SEG/images'
        train_mask_dir = '../dataset/Kvasir-SEG/masks'
        train_img_ids_ = glob(os.path.join(train_dir, '*' + '.jpg'))
        train_img_ids_ = [os.path.splitext(os.path.basename(p))[0] for p in train_img_ids_]
    elif config['dataset'] == 'kvasir-instrument':
        train_dir = '../dataset/kvasir-instrument/images'
        train_mask_dir = '../dataset/kvasir-instrument/masks'
        train_img_ids_ = glob(os.path.join(train_dir, '*' + '.jpg'))
        train_img_ids_ = [os.path.splitext(os.path.basename(p))[0] for p in train_img_ids_]
    elif config['dataset'] == 'COVID-19':
        train_dir = './dataset/COVID-19/images'
        train_mask_dir = './dataset/COVID-19/masks'
        # train_dir = './cvc-clinicdb/PNG/Original'
        # train_mask_dir = './cvc-clinicdb/PNG/Ground_Truth'
        train_img_ids_ = glob(os.path.join(train_dir, '*' + '.png'))
        train_img_ids_ = [os.path.splitext(os.path.basename(p))[0] for p in train_img_ids_]
    elif config['dataset'] == 'ICF':
        train_dir = '../dataset/ICF/images'
        train_mask_dir = '../dataset/ICF/masks'
        train_img_ids_ = glob(os.path.join(train_dir, '*' + '.jpeg'))
        train_img_ids_ = [os.path.splitext(os.path.basename(p))[0] for p in train_img_ids_]
    
    ##train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)
    if config['dataset'] == 'ISIC2018':
        config['input_h'] = 416
        config['input_w'] = 416
        config['img_ext'] = '.jpg'
        config['mask_ext'] = '.png'
    elif config['dataset'] == 'ISIC2018NEW2':
        config['input_h'] = 416
        config['input_w'] = 416
        config['img_ext'] = '.jpg'
        config['mask_ext'] = '.png'
    elif config['dataset'] == 'BUSI':
        config['input_h'] = 384
        config['input_w'] = 384
        config['img_ext'] = '.png'
        config['mask_ext'] = '.png'
    elif config['dataset'] == 'cvc-clinicdb':
        config['input_h'] = 256
        config['input_w'] = 256
        config['img_ext'] = '.png'
        config['mask_ext'] = '.png'
    elif config['dataset'] == 'Kvasir-SEG':
        config['input_h'] = 256
        config['input_w'] = 256
        config['img_ext'] = '.jpg'
        config['mask_ext'] = '.jpg'
    elif config['dataset'] == 'kvasir-instrument':
        config['input_h'] = 256
        config['input_w'] = 256
        config['img_ext'] = '.jpg'
        config['mask_ext'] = '.png'
    elif config['dataset'] == 'COVID-19':
        config['input_h'] = 256
        config['input_w'] = 256
        config['img_ext'] = '.png'
        config['mask_ext'] = '.png'
    elif config['dataset'] == 'ICF':
        config['input_h'] = 512
        config['input_w'] = 512
        config['img_ext'] = '.jpeg'
        config['mask_ext'] = '.png'
    elif config['dataset'] == 'Covid_Infection':
        config['input_h'] = 224
        config['input_w'] = 224
    
    if config['arch'] == 'SwinUnet':
        config['input_h'] = 224
        config['input_w'] = 224
    
    train_transform = Compose([
        RandomRotate90(),
        Flip(),
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ], is_check_shapes=False)

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ], is_check_shapes=False)
    if config['dataset'] == 'cvc-clinicdb' or config['dataset'] == 'COVID-19' or config['dataset'] == 'ICF':
        dataset = Dataset(
            img_ids=train_img_ids_,
            img_dir=train_dir,
            mask_dir=train_mask_dir,
            img_ext=config['img_ext'],
            mask_ext=config['mask_ext'],
            num_classes=config['num_classes'],
            transform=train_transform)
        n_val = int(len(dataset) * 0.1)
        n_train = len(dataset) - n_val
        
        train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(1))
    elif config['dataset'] == 'Kvasir-SEG':
        dataset = Dataset(
            img_ids=train_img_ids_,
            img_dir=train_dir,
            mask_dir=train_mask_dir,
            img_ext=config['img_ext'],
            mask_ext=config['mask_ext'],
            num_classes=config['num_classes'],
            transform=train_transform)
        n_val = int(len(dataset) * 0.12)
        n_train = len(dataset) - n_val
        print(n_val)
        print(n_train)
        train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    elif config['dataset'] == 'kvasir-instrument':
        dataset = Dataset(
            img_ids=train_img_ids_,
            img_dir=train_dir,
            mask_dir=train_mask_dir,
            img_ext=config['img_ext'],
            mask_ext=config['mask_ext'],
            num_classes=config['num_classes'],
            transform=train_transform)
        n_val = int(len(dataset) * 0.2)
        n_train = len(dataset) - n_val
        print(n_val)
        print(n_train)
        train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    elif config['dataset'] == 'Covid_Infection':
        train_transforms = A.Compose(
            [
                A.Resize(height=config['input_h'], width=config['input_w']),
                A.Rotate(limit=35, p=1.0),
                A.HorizontalFlip(p=0.2),
                A.VerticalFlip(p=0.2),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )
        ######
        val_transforms = A.Compose(
            [
                A.Resize(height=config['input_h'], width=config['input_w']),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )
        train_dataset = Data_loaderVE(
            root='/media/laboratory/wangjie/dataset/Covid_Infection/'
            ,train = 'Training_data.pt'
            ,transform = train_transforms 
        )

        val_dataset = Data_loaderVE(
            root='/media/laboratory/wangjie/dataset/Covid_Infection/'
            ,train = 'Validation_data.pt'
            ,transform = val_transforms
        )
    else:
        train_dataset = Dataset(
            img_ids=train_img_,
            img_dir=train_dir,
            mask_dir=train_mask_dir,
            img_ext=config['img_ext'],
            mask_ext=config['mask_ext'],
            num_classes=config['num_classes'],
            transform=train_transform)
        val_dataset = Dataset(
            img_ids=val_img_,
            img_dir=val_dir,
            mask_dir=val_mask_dir,
            img_ext=config['img_ext'],
            mask_ext=config['mask_ext'],
            num_classes=config['num_classes'],
            transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    # create model
    if config['arch'] == 'UNext':
        model = archsSEMlp.__dict__[config['arch']](config['num_classes'],
                                        config['input_channels'],
                                        config['deep_supervision'])
    elif config['arch'] == 'Test_SACA_UNet':
        model = archsSEMlp.__dict__[config['arch']](config['num_classes'],
                                               config['input_channels'],
                                               config['deep_supervision'])
    elif config['arch'] == 'SkipConnect_SACA_UNet':
        model = archsSEMlp.__dict__[config['arch']](config['num_classes'],
                                               config['input_channels'],
                                               config['deep_supervision'])
    elif config['arch'] == 'SkipConnect_SACA_UNet_resize':
        model = archsSEMlp.__dict__[config['arch']](config['num_classes'],
                                               config['input_channels'],
                                               config['deep_supervision'])
    elif config['arch'] == 'SkipConnect_SACA_UNet_crop_resize':
        model = archsSEMlp.__dict__[config['arch']](config['num_classes'],
                                               config['input_channels'],
                                               config['deep_supervision'])
    elif config['arch'] == 'MPSAttention_two':
        model = archsSEMlp.__dict__[config['arch']](config['num_classes'],
                                               config['input_channels'],
                                               config['deep_supervision'])
    elif config['arch'] == 'EncodeNum_SACA_UNet':
        model = archsSEMlp.__dict__[config['arch']](config['num_classes'],
                                               config['input_channels'],
                                               config['deep_supervision'])                                          
    elif config['arch'] == 'MulEncode_SACA_UNet':
        model = archsSEMlp.__dict__[config['arch']](config['num_classes'],
                                               config['input_channels'],
                                               config['deep_supervision'])
    elif config['arch'] == 'Dense_Test_SACA_UNet':
        model = archsSEMlp.__dict__[config['arch']](config['num_classes'],
                                                    config['input_channels'],
                                                    config['deep_supervision'])
    elif config['arch'] == 'SACA_UNet':
        model = archsSEMlp.__dict__[config['arch']](config['num_classes'],
                                               config['input_channels'],
                                               config['deep_supervision'])
    elif config['arch'] == 'SACA_UNet_Att':
        model = archsSEMlp.__dict__[config['arch']](config['num_classes'],
                                               config['input_channels'],
                                               config['deep_supervision'])
    elif config['arch'] == 'QuinNet':
        model = archsSEMlp.__dict__[config['arch']](config['num_classes'],
                                               config['input_channels'],
                                               config['deep_supervision'])
    elif config['arch'] == 'PAttUNet':
        model = PAttUNet(input_channels=3, num_classes = 1)
    elif config['arch'] == 'UNetplus':
        model = UNetplus(input_channels=3, num_classes=1)
    elif config['arch'] == 'AttUNet':
        model = AttUNet(input_channels=3, num_classes=1)
    elif config['arch'] == 'UNet':
        model = UNet(input_channels=3, num_classes=config['num_classes'])
    elif config['arch'] == 'SwinUnet':
        model = archsSEMlp.__dict__[config['arch']](num_classes=config['num_classes'],
                                               input_channels=config['input_channels'],
                                               img_size=config['input_h'])
    elif config['arch'] == 'SwinUNetR':
        model = archsSEMlp.__dict__[config['arch']](input_channels=config['input_channels'],
                                               out_channels=1,
                                               img_size=config['input_h'])
    elif config['arch'] == 'nnFormer':
        model = archsSEMlp.__dict__[config['arch']](crop_size=(1, config['input_h'], config['input_h']))
    elif config['arch'] == 'Resnet34_Unet':
        model = Resnet34_Unet(in_channel=3, out_channel=1, pretrained=True)
    elif config['arch'] == 'MTUNet':
        model = MTUNet(num_classes=1)
    elif config['arch'] == 'FCT':
        model = FCT()
    model = model.to(device)
    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError
    ###########打印信息#####################
    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)
    ########################################
    print('======加载数据集完成========')
    print('train_loader: ', len(train_loader))
    print('val_loader: ', len(val_loader))
    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
        ('val_Acc', []),
        ('val_mAcc', []),
        ('val_mIoU', []),
        ('val_recall', []),
        ('val_spec', [])
    ])

    best_iou = 0
    trigger = 0
    
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f - Acc %.4f - mAcc %.4f - mIoU %.4f - recall %.4f - spec %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou'], val_log['Acc'], val_log['mAcc'], val_log['mIoU'],val_log['recall'] ,val_log['spec']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])
        log['val_Acc'].append(val_log['Acc'])
        log['val_mAcc'].append(val_log['mAcc'])
        log['val_mIoU'].append(val_log['mIoU'])
        log['val_recall'].append(val_log['recall'])
        log['val_spec'].append(val_log['spec'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)
        writer.add_scalar('info/lr', config['lr'], epoch)
        writer.add_scalar('info/train_loss', train_log['loss'], epoch)
        writer.add_scalar('info/train_iou', train_log['iou'], epoch)
        writer.add_scalar('info/val_loss', val_log['loss'], epoch)
        writer.add_scalar('info/val_iou', val_log['iou'], epoch)
        writer.add_scalar('info/val_dice', val_log['dice'], epoch)
        writer.add_scalar('info/val_recall', val_log['recall'], epoch)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'models/%s/model.pth' %
                       config['name'])
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()
    writer.close()

if __name__ == '__main__':
    main()
    
