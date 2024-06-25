from dataset.semi import SemiDataset
from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.pspnet import PSPNet
from utils.utils import count_params, MeanIOU, meanIOU, DiceCoefficient, color_map
from utils.DICELOSS import DiceLoss
from utils.loss_file import save_loss

import argparse
from copy import deepcopy
import numpy as np
import os
from PIL import Image
import torch
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm


MODE = None

def parse_args():
    parser = argparse.ArgumentParser(description='ST and ST++ Framework')

    # basic settings
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--dataset', type=str, choices=['pascal', 'cityscapes', 'kidney'], default='kidney')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--crop-size', type=int, default=None)
    parser.add_argument('--backbone', type=str, choices=['resnet18', 'resnet50', 'resnet101'], default='resnet18')
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--model', type=str, choices=['deeplabv3plus', 'pspnet', 'deeplabv2'],
                        default='deeplabv3plus')

    # semi-supervised settings
    parser.add_argument('--labeled-id-path', type=str, required=True)
    parser.add_argument('--unlabeled-id-path', type=str, required=True)
    parser.add_argument('--validation-id-path', type=str, required=True)
    parser.add_argument('--pseudo-mask-path', type=str, required=True)

    parser.add_argument('--save-model-path', type=str, required=True)
    parser.add_argument('--loss-file-path', type=str, required=True)

    # arguments for ST++
    parser.add_argument('--reliable-id-path', type=str)
    parser.add_argument('--plus', dest='plus', default=False, action='store_true',
                        help='whether to use ST++')

    args = parser.parse_args()
    return args


def main(args):
    global MODE
    MODE = 'train'

    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
    if not os.path.exists(args.pseudo_mask_path):
        os.makedirs(args.pseudo_mask_path)
    if args.plus and args.reliable_id_path is None:
        exit('Please specify reliable-id-path in ST++.')

    # criterion = CrossEntropyLoss(ignore_index=255)
    criterion = DiceLoss()

    valset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size, args.validation_id_path)    
    valloader = DataLoader(valset, batch_size=4, shuffle=True, pin_memory=True, num_workers=4, drop_last=True)

    # <====================== Supervised training with labeled images (SupOnly) ======================>
    print('\n================> Total stage 1/%i: '
          'Supervised training on labeled images (SupOnly)' % (6 if args.plus else 3))

    trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size, args.labeled_id_path)
    trainset.ids = 2 * trainset.ids if len(trainset.ids) < 200 else trainset.ids
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=16, drop_last=True)

    model, optimizer = init_basic_elems(args)
    print('\nParams: %.1fM' % count_params(model))

    best_model, checkpoints = train(model, trainloader, valloader, criterion, optimizer, args,     step=f'st{"++" if args.plus else ""}_supervised_labeled')

    """
        ST framework without selective re-training
    """
    if not args.plus:
        # <============================= Pseudo label all unlabeled images =============================>
        print('\n\n\n================> Total stage 2/3: Pseudo labeling all unlabeled images')

        dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, args.unlabeled_id_path)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

        label(best_model, dataloader, args)

        # <======================== Re-training on labeled and unlabeled images ========================>
        print('\n\n\n================> Total stage 3/3: Re-training on labeled and unlabeled images')

        MODE = 'semi_train'

        trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size,
                               args.labeled_id_path, args.unlabeled_id_path, args.pseudo_mask_path)
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                 pin_memory=True, num_workers=16, drop_last=True)

        model, optimizer = init_basic_elems(args)

        train(model, trainloader, valloader, criterion, optimizer, args, step='st-semi-supervised')

        return

    """
        ST++ framework with selective re-training
    """
    # <===================================== Select Reliable IDs =====================================>
    print('\n\n\n================> Total stage 2/6: Select reliable images for the 1st stage re-training')

    dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, args.unlabeled_id_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    select_reliable(checkpoints, dataloader, args)

    # <================================ Pseudo label reliable images =================================>
    print('\n\n\n================> Total stage 3/6: Pseudo labeling reliable images')

    cur_unlabeled_id_path = os.path.join(args.reliable_id_path, 'reliable_ids.txt')
    dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, cur_unlabeled_id_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    label(best_model, dataloader, args)

    # <================================== The 1st stage re-training ==================================>
    print('\n\n\n================> Total stage 4/6: The 1st stage re-training on labeled and reliable unlabeled images')

    MODE = 'semi_train'

    trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size,
                           args.labeled_id_path, cur_unlabeled_id_path, args.pseudo_mask_path)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=16, drop_last=True)

    model, optimizer = init_basic_elems(args)

    best_model = train(model, trainloader, valloader, criterion, optimizer, args, step='re-training-1st')

    # <=============================== Pseudo label unreliable images ================================>
    print('\n\n\n================> Total stage 5/6: Pseudo labeling unreliable images')

    cur_unlabeled_id_path = os.path.join(args.reliable_id_path, 'unreliable_ids.txt')
    dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, cur_unlabeled_id_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    label(best_model, dataloader, args)

    # <================================== The 2nd stage re-training ==================================>
    print('\n\n\n================> Total stage 6/6: The 2nd stage re-training on labeled and all unlabeled images')

    trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size,
                           args.labeled_id_path, args.unlabeled_id_path, args.pseudo_mask_path)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=16, drop_last=True)

    model, optimizer = init_basic_elems(args)

    train(model, trainloader, valloader, criterion, optimizer, args, step='re-training-2st')


def init_basic_elems(args):
    model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2}
    model = model_zoo[args.model](args.backbone, args.num_classes)

    head_lr_multiple = 10.0
    if args.model == 'deeplabv2':
        assert args.backbone == 'resnet101'
        model.load_state_dict(torch.load('pretrained/deeplabv2_resnet101_coco_pretrained.pth'))
        head_lr_multiple = 1.0

    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': args.lr},
                     {'params': [param for name, param in model.named_parameters()
                                 if 'backbone' not in name],
                      'lr': args.lr * head_lr_multiple}],
                    lr=args.lr, momentum=0.9, weight_decay=1e-4)

    model = DataParallel(model).cuda()

    return model, optimizer


def train(model, trainloader, valloader, criterion, optimizer, args, step=""):
    iters = 0
    total_iters = len(trainloader) * args.epochs

    previous_best = 0.0

    global MODE

    if MODE == 'train':
        checkpoints = []

    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.4f\t\t\t\t\t previous best = %.2f" %
              (epoch, optimizer.param_groups[0]["lr"], previous_best))

        model.train()
        total_t_loss = 0.0
        metric_t_miou = MeanIOU()
        metric_t_dice = DiceCoefficient()
        tbar = tqdm(trainloader)

        for i, (img, mask) in enumerate(tbar):
            img, mask = img.cuda(), mask.cuda()

            pred = model(img)
            loss = criterion(pred, mask)
            
            metric_t_miou.add_batch(pred, mask)
            metric_t_dice.add_batch(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_t_loss += loss.item()

            iters += 1
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 1.0 if args.model == 'deeplabv2' else lr * 10.0

            tbar.set_description('Loss: %.3f' % (total_t_loss / (i + 1)))



        model.eval()
        tbar = tqdm(valloader)
        total_v_loss = 0
        metric_v_miou = MeanIOU()
        metric_v_dice = DiceCoefficient()

        with torch.no_grad():
            for i, (img, mask) in enumerate(tbar):
                img = img.cuda()
                mask = mask.cuda()

                pred = model(img)

                loss = criterion(pred, mask)
                metric_v_miou.add_batch(pred, mask)
                metric_v_dice.add_batch(pred, mask)
                total_v_loss += loss.item()
                
                mIOU = metric_v_miou.evaluate()
                tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))

        avg_train_loss = total_t_loss / len(trainloader)
        avg_train_miou = metric_t_miou.evaluate()
        avg_train_dice = metric_t_dice.evaluate()
        avg_val_loss = total_v_loss / len(valloader)
        avg_val_miou = metric_v_miou.evaluate()
        avg_val_dice = metric_v_dice.evaluate()

        save_loss(
            t_loss=avg_train_loss, 
            t_miou=avg_train_miou,    
            t_dice=avg_train_dice,
            v_loss=avg_val_loss, 
            v_miou=avg_val_miou,    
            v_dice=avg_val_dice,
            filename= f'{args.loss_file_path}/{step}_loss.csv'
        )
        mIOU *= 100.0
        if mIOU > previous_best:
            if previous_best != 0:
                os.remove(os.path.join(args.save_model_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, previous_best)))
            previous_best = mIOU
            torch.save(model.module.state_dict(),
                       os.path.join(args.save_model_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, mIOU)))

            best_model = deepcopy(model)

        if MODE == 'train' and ((epoch + 1) in [args.epochs // 3, args.epochs * 2 // 3, args.epochs]):
            checkpoints.append(deepcopy(model))

    torch.save(model.module.state_dict(),f"{args.save_model_path}/{step}_model_weight.pth")
    if MODE == 'train':
        return best_model, checkpoints

    return best_model


def select_reliable(models, dataloader, args):
    if not os.path.exists(args.reliable_id_path):
        os.makedirs(args.reliable_id_path)

    for i in range(len(models)):
        models[i].eval()
    tbar = tqdm(dataloader)

    id_to_reliability = []

    with torch.no_grad():
        for img, mask, id in tbar:
            img = img.cuda()

            preds = []
            for model in models:
                preds.append(torch.argmax(model(img), dim=1).cpu().numpy())

            mIOU = []
            for i in range(len(preds) - 1):
                metric = meanIOU(num_classes=2)
                metric.add_batch(preds[i], preds[-1])
                mIOU.append(metric.evaluate()[-1])

            reliability = sum(mIOU) / len(mIOU)
            id_to_reliability.append((id[0], reliability))

    id_to_reliability.sort(key=lambda elem: elem[1], reverse=True)
    with open(os.path.join(args.reliable_id_path, 'reliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[:len(id_to_reliability) // 2]:
            f.write(elem[0] + '\n')
    with open(os.path.join(args.reliable_id_path, 'unreliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[len(id_to_reliability) // 2:]:
            f.write(elem[0] + '\n')


# def label(model, dataloader, args):
#     model.eval()
#     tbar = tqdm(dataloader)

#     metric = meanIOU(num_classes=2)
#     cmap = color_map(args.dataset)

#     with torch.no_grad():
#         for img, mask, id in tbar:
#             img = img.cuda()
#             pred = model(img, True)
#             pred = torch.argmax(pred, dim=1).cpu()

#             metric.add_batch(pred.numpy(), mask.numpy())
#             mIOU = metric.evaluate()[-1]

#             pred = Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8), mode='P')
#             pred.putpalette(cmap)

#             pred.save('%s/%s' % (args.pseudo_mask_path, os.path.basename(id[0].split(' ')[1])))

#             tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))

def label(model, dataloader, args):
    model.eval()
    tbar = tqdm(dataloader)
    class_colors = [
        [0, 0, 0],  # 类别 0 的颜色为黑色
        [255, 0, 0]  # 类别 1 的颜色为红色
    ]
    # 创建调色板
    cmap = np.array(class_colors, dtype=np.uint8)#.flatten()
    with torch.no_grad():
        for img,mask,id in tbar:
            img = img.cuda()
            pred = model(img, True)
            pred = torch.argmax(pred, dim=1).cpu()
            pred = Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8), mode='P')
            # 2023.6.21 modify
            pred.putpalette(cmap)
            pred.save('%s/%s' % (args.pseudo_mask_path, os.path.basename(id[0].replace('.jpg','.png'))))

if __name__ == '__main__':
    args = parse_args()

    if args.epochs is None:
        args.epochs = {'pascal': 80, 'cityscapes': 240, 'kidney': 60}[args.dataset]
    if args.lr is None:
        args.lr = {'pascal': 0.001, 'cityscapes': 0.004, 'kidney': 0.001}[args.dataset] / 16 * args.batch_size
    if args.crop_size is None:
        args.crop_size = {'pascal': 321, 'cityscapes': 721, 'kidney': 224}[args.dataset]

    print()
    print(args)

    main(args)
