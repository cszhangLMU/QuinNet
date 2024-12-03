import argparse
import os
from glob import glob

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from archsSEMlp import  MTUNet, UNetplus, UNet, AttUNet, SwinUnet, SwinUNetR
import archsSEMlp
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
from albumentations import RandomRotate90,Resize
import time
from archsSEMlp import UNext
from torch.utils.data import random_split


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='ISIC2018_UNext_woDS',
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
   
    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True
   
    # Data loading code
    ##img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    ##img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    #val Data loading code
    ##train_img_ids_ = glob(os.path.join('dataset', config['dataset'], 'Training_Input', '*' + '.jpg'))
    
    # Valition data loading code
    val_dir = ''
    val_mask_dir = ''
    val_img_ = []
    if config['dataset'] == 'ISIC2018':
        # Valition data loading code
        val_img_ids_ = glob(os.path.join('./dataset/ISIC2018/Validation_Input', '*' + '.jpg'))
        val_img_ids_ = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids_]
        val_img_ = val_img_ids_
        val_dir = './dataset/ISIC2018/Validation_Input'
        val_mask_dir = os.path.join('./dataset/ISIC2018', 'Validation_GroundTruth')
    elif config['dataset'] == 'BUSI':
        val_img_ids_ = glob(os.path.join('./dataset/BUSI/val', '*' + '.png'))
        val_img_ids_ = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids_]
        
        for p in val_img_ids_:
            m = p.split('_')
            if m[-1] != 'mask':
                val_img_.append(m[0])
        val_dir = './dataset/BUSI/val'
        val_mask_dir = val_dir
        config['img_ext'] = '.png'
    elif config['dataset'] == 'cvc-clinicdb':
        # train_dir = '../dataset/CVC-ClinicDB/PNG/Original/'
        # train_mask_dir = '../dataset/CVC-ClinicDB/PNG/Ground_Truth/'
        train_dir = './dataset/CVC-ClinicDB/PNG/Original'
        train_mask_dir = './dataset/CVC-ClinicDB/PNG/Ground_Truth'
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
    if config['dataset'] == 'ISIC2018':
        config['input_h'] = config['input_w'] = 512
    elif config['dataset'] == 'BUSI':
        config['input_h'] = config['input_w'] = 256
    elif config['dataset'] == 'cvc-clinicdb':
        config['input_h'] = 256
        config['input_w'] = 256
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
    if config['arch'] == 'SwinUnet':
        config['input_h'] = 224
        config['input_w'] = 224
    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ], is_check_shapes=False)
    if config['dataset'] == 'cvc-clinicdb' or config['dataset'] == 'COVID-19':
        dataset = Dataset(
            img_ids=train_img_ids_,
            img_dir=train_dir,
            mask_dir=train_mask_dir,
            img_ext=config['mask_ext'],
            mask_ext=config['mask_ext'],
            num_classes=config['num_classes'],
            transform=val_transform)
        n_val = int(len(dataset))
        n_train = len(dataset) - n_val
        print(n_val)
        print(n_train)
        train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    elif config['dataset'] == 'Kvasir-SEG':
        dataset = Dataset(
            img_ids=train_img_ids_,
            img_dir=train_dir,
            mask_dir=train_mask_dir,
            img_ext=config['img_ext'],
            mask_ext=config['mask_ext'],
            num_classes=config['num_classes'],
            transform=val_transform)
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
            transform=val_transform)
        n_val = int(len(dataset) * 0.1)
        n_train = len(dataset) - n_val
        print(n_val)
        print(n_train)
        train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    else:
        val_dataset = Dataset(
            img_ids=val_img_,
            img_dir=val_dir,
            mask_dir=val_mask_dir,
            img_ext=config['img_ext'],
            mask_ext=config['mask_ext'],
            num_classes=config['num_classes'],
            transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    print("=> creating model %s" % config['arch'])
    
    if config['arch'] == 'UNet':
        model = UNet(input_channels=3, num_classes=1)
    elif config['arch'] == 'UNetplus':
        model = UNetplus(input_channels=3, num_classes=1)
    elif config['arch'] == 'AttUNet':
        model = AttUNet(input_channels=3, num_classes=1)
    elif config['arch'] == 'MTUNet':
        model = MTUNet(num_classes=1)
    elif config['arch'] == 'SwinUnet':
        model = archsSEMlp.__dict__[config['arch']](num_classes=config['num_classes'],
                                               input_channels=config['input_channels'],
                                               img_size=config['input_h'])
    elif config['arch'] == 'SwinUNetR':
        model = SwinUNetR(input_channels=config['input_channels'], out_channels=1, img_size=config['input_h'])
    else:
        model = archsSEMlp.__dict__[config['arch']](config['num_classes'],
                                                    config['input_channels'],
                                                    config['deep_supervision'])

    model = model.cuda()

    print(config['name'])
    model.load_state_dict(torch.load('models/%s/best_model.pth' % config['name']))
    model.eval()

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    gput = AverageMeter()
    cput = AverageMeter()

    count = 0
    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()
            model = model.cuda()
            # compute output
            if config['arch'] == 'MPSAttention_two':
                mspOutput1,mspOutput2,mspOutput3,mspOutput4,output = model(input)
            else:
                output = model(input)


            iou,dice = iou_score(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()
            output[output>=0.5]=1
            output[output<0.5]=0

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))

    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
