import models
from models.readout_nets.center_bias_layer import CenterBiasLayer
from models.readout_nets.equator_bias_layer import EquatorBiasLayer
from utils.meter import MeterStorage, AverageMeter
from utils.loaders import TrainLoader
from utils.checkpoint import save_checkpoint
from utils.loss import KLD
from utils import np_transforms
from utils.progress_bar import progress_bar
from utils.dataset import SaliencyDataset
from utils.dataset import MultiScaleSaliencyDataset

from odimapping.unify import cropping
from models.fusion_layer import FusionLayer
from models.attention_layer import Attention
from models.attention_layer import AttentionV2
from models.attention_layer import AttentionWithFeatures
from models.attention_layer import AttentionV2WithFeatures
from models.composite import composite_avg
from models.composite import composite_max

import argparse
from datetime import datetime
import os

import imageio
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.ndimage import zoom

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/sample.yaml',
                        help='config file')
    return parser

# def extract_features(model, target, inputs):
#     features = None
#     def forward_hook(module, inputs, outputs):
#         global features
#         features = inputs[0].detach().clone()

#     handle = target.register_forward_hook(forward_hook)
#     output = model(inputs)
#     handle.remove()
#     return [output, features]

def train(epoch, optimizer, model, center_bias_layer, equator_bias_layer, fusion_layer, attention_layer, train_dataset,
          train_loader, criterion, use_center_bias_layer=False, use_equator_bias_layer=False, use_fusion_layer=False,
          use_attention_layer=False, use_attention_features=False, arch='densesalbi3'):
    def print_model(module, name="model", depth=0):
        if len(list(module.named_children())) == 0:
            print(f"{' ' * depth} {name}: {module}")
        else:
            print(f"{' ' * depth} {name}: {type(module)}")

        for child_name, child_module in module.named_children():
            if isinstance(module, torch.nn.Sequential):
                child_name = f"{name}[{child_name}]"
            else:
                child_name = f"{name}.{child_name}"
            print_model(child_module, child_name, depth + 1)

    # To confirm the model
    # print_model(model)

    meters = MeterStorage()
    meters.add('Loss')
    model.train()

    # if use_attention_features and arch != 'deepgaze2e':
    #     def forward_hook(module, inputs, outputs):
    #         global features
    #         features = inputs[0]

    #     # To get final-layer features in densenet161
    #     if arch == 'dpnsal131_dilation_multipath':
    #         target_module = model.readout_net
    #     else:
    #         target_module = model.readout_net[0]
    #     handle = target_module.register_forward_hook(forward_hook)

    if use_center_bias_layer:
        center_bias_layer.train()
    if use_equator_bias_layer:
        equator_bias_layer.train()
    if use_fusion_layer:
        fusion_layer.train()
    if use_attention_layer:
        attention_layer.train()
    for i, (input_image_list, gt_map_list, view_angle_list, path) in enumerate(train_loader):
        outputs = []
        loss = 0
        min_angle_features = None

        for j in range(len(view_angle_list)):
            input_image = input_image_list[j]
            input_image = input_image.to(DEVICE)
            gt_map = gt_map_list[j].to(DEVICE)

            if use_attention_features:
                if arch == 'deepgaze2e':
                    centerbias_tensor = torch.tensor(np.zeros((1, input_image.shape[2], input_image.shape[3]))).to(DEVICE)
                    [log_density, features] = model(input_image, centerbias_tensor)
                    output = (torch.exp(log_density)).float()
                    features = features.float()
                    features = features.detach() # do not feedback grad. https://qiita.com/ground0state/items/15f218ab89121d66b462
                else:
                    [output, features] = model(input_image)
                    # if arch == 'dpnsal131_dilation_multipath':
                    #     target_module = model.readout_net
                    # else:
                    #     target_module = model.readout_net[0]
                    # [output, features] = extract_features(model, target_module, input_image)
                    # print(features.shape)
            else:
                output = model(input_image)

            if use_center_bias_layer:
                output = center_bias_layer(output)
            if use_equator_bias_layer:
                file_name_split = path[0].split('_')
                extract_number = int(file_name_split[-1])
                num_horizontal = train_dataset.num_horizontal
                if extract_number == 0:
                    eqbl_idx = 0
                elif extract_number == 1:
                    eqbl_idx = train_dataset.num_vertical - 1
                else:
                    eqbl_idx = np.ceil((extract_number - 1) / num_horizontal).astype(np.int)
                output = equator_bias_layer(output, eqbl_idx)

            loss += criterion(output, gt_map)

            if j != 0:
                output = cropping(view_angle_list[0], view_angle_list[j], output)

            if j == 0 and use_attention_features:
                min_angle_features = features

            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)

        if use_fusion_layer:
            output = fusion_layer(outputs)
        if use_attention_layer:
            if use_attention_features:
                output = attention_layer(outputs, min_angle_features)  # for AttentionV3
            else:
                output = attention_layer(outputs)

        gt_map = gt_map_list[0].to(DEVICE)
        loss += criterion(output, gt_map)
        meters.update('Loss', loss.item(), input_image_list[0].size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar(i, len(train_loader), '[Train] Epoch: {}, Loss: {:.3f}'.format(epoch, meters.Loss.avg))
        train_loss = meters.Loss.avg

    return train_loss


def val(epoch, model, center_bias_layer, equator_bias_layer, fusion_layer, attention_layer, val_dataset, val_loader,
        criterion, use_center_bias_layer=False, use_equator_bias_layer=False, use_fusion_layer=False,
        use_attention_layer=False, use_attention_features=False, arch='densesalbi3'):
    meters = MeterStorage()
    meters.add('Loss')

    # if use_attention_features and arch != 'deepgaze2e':
    #     def forward_hook(module, inputs, outputs):
    #         global features
    #         features = inputs[0]

    #     # To get final-layer features in densenet161
    #     if arch == 'dpnsal131_dilation_multipath':
    #         target_module = model.readout_net
    #     else:
    #         target_module = model.readout_net[0]
    #     handle = target_module.register_forward_hook(forward_hook)

    with torch.no_grad():
        for i, (input_image_list, gt_map_list, view_angle_list, path) in enumerate(val_loader):

            outputs = []
            loss = 0
            min_angle_features = None

            for j in range(len(view_angle_list)):

                input_image = input_image_list[j]
                input_image = input_image.to(DEVICE)
                gt_map = gt_map_list[j].to(DEVICE)

                if use_attention_features:
                    if arch == 'deepgaze2e':
                        centerbias_tensor = torch.tensor(np.zeros((1, input_image.shape[2], input_image.shape[3]))).to(DEVICE)
                        [log_density, features] = model(input_image, centerbias_tensor)
                        output = (torch.exp(log_density)).float()
                        features = features.float()
                        features = features.detach() # do not feedback grad. https://qiita.com/ground0state/items/15f218ab89121d66b462
                    else:
                        [output, features] = model(input_image)
                        # if arch == 'dpnsal131_dilation_multipath':
                        #     target_module = model.readout_net
                        # else:
                        #     target_module = model.readout_net[0]
                        # [output, features] = extract_features(model, target_module, input_image)
                        # print(features.shape)
                else:
                    output = model(input_image)

                if use_center_bias_layer:
                    output = center_bias_layer(output)
                if use_equator_bias_layer:
                    file_name_split = path[0].split('_')
                    extract_number = int(file_name_split[-1])
                    num_horizontal = val_dataset.num_horizontal
                    if extract_number == 0:
                        eqbl_idx = 0
                    elif extract_number == 1:
                        eqbl_idx = val_dataset.num_vertical - 1
                    else:
                        eqbl_idx = np.ceil((extract_number - 1) / num_horizontal).astype(np.int)
                    output = equator_bias_layer(output, eqbl_idx)

                loss += criterion(output, gt_map)
                if j != 0:
                    output = cropping(view_angle_list[0], view_angle_list[j], output)

                outputs.append(output)

                if j == 0 and use_attention_features:
                    min_angle_features = features

            outputs = torch.cat(outputs, dim=1)

            if use_fusion_layer:
                output = fusion_layer(outputs)
            if use_attention_layer:
                if use_attention_features:
                    output = attention_layer(outputs, min_angle_features)  # for AttentionWithFeatures
                else:
                    output = attention_layer(outputs)

            gt_map = gt_map_list[0].to(DEVICE)
            loss += criterion(output, gt_map)
            meters.update('Loss', loss.item(), input_image_list[0].size(0))
            progress_bar(i, len(val_loader), '[Validation] Epoch: {}, Loss: {:.3f}'.format(epoch, meters.Loss.avg))
            val_loss = meters.Loss.avg
    return val_loss


def main(cfg):
    print('-----------------------')
    start_time_stamp = '{0:%Y%m%d-%H%M%S}'.format(datetime.now())
    log_path = None
    print('START TIME : {}'.format(start_time_stamp))

    use_equator_bias_layer = cfg['SETTING']['BIAS']['USE_EQUATOR_BIAS_LAYER']
    use_center_bias_layer = cfg['SETTING']['BIAS']['USE_CENTER_BIAS_LAYER']
    use_constant_equator_bias = cfg['SETTING']['BIAS']['USE_CONSTANT_EQUATOR_BIAS']
    batch_size = cfg['SETTING']['BATCH_SIZE']
    use_fusion_layer = cfg['SETTING']['BIAS']['USE_FUSION_LAYER']
    use_attention_layer = cfg['SETTING']['BIAS']['USE_ATTENTION_LAYER']
    use_basic_attention = cfg['SETTING']['BIAS']['USE_BASIC_ATTENTION_LAYER']
    use_basic_attention_v2 = cfg['SETTING']['BIAS']['USE_BASIC_ATTENTION_V2_LAYER']
    use_basic_attention_with_features = cfg['SETTING']['BIAS']['USE_BASIC_ATTENTION_WITH_FEATURE']

    model_names = sorted(name for name in models.__dict__
                         if not name.startswith("_") and callable(models.__dict__[name]))

    # Init MultiScaleSaliencyDatasetClass
    # Loading config, reading images, determining file format, and so on.
    # Train
    train_dataset = MultiScaleSaliencyDataset(
        cfg['DATA']['DATASET_CFG_PATH'],
        train=True,
        target_type=['distribution'],
        transform=np_transforms.ResizeToInnerRectangle(rec_long_side=cfg['DATA']['INPUT_IMAGE_WIDTH'],
                                                       rec_short_side=cfg['DATA']['INPUT_IMAGE_HEIGHT']),
        post_simul_transform=np_transforms.ToTensor(),
    )
    # Init MultiScaleSaliencyDatasetClass
    # Loading config, reading images, determining file format, and so on.
    # Val
    val_dataset = MultiScaleSaliencyDataset(
        cfg['DATA']['DATASET_CFG_PATH'],
        val=True,
        target_type=['distribution'],
        transform=np_transforms.ResizeToInnerRectangle(rec_long_side=cfg['DATA']['INPUT_IMAGE_WIDTH'],
                                                       rec_short_side=cfg['DATA']['INPUT_IMAGE_HEIGHT']),
        post_simul_transform=np_transforms.ToTensor(),
    )
    if len(val_dataset) == 0:
        has_val = False
    else:
        has_val = True
    print('DATASET : {}'.format(train_dataset.dataset_name))
    if use_equator_bias_layer:
        batch_size = 1
        print('using equator bias layer -> batch_size = 1')
    print('BATCH SIZE : {}'.format(batch_size))

    train_loader = TrainLoader(train_dataset, batch_size=batch_size, workers=cfg['SETTING']['WORKERS'])
    if has_val:
        val_loader = TrainLoader(val_dataset, batch_size=batch_size, workers=cfg['SETTING']['WORKERS'])

    # get model
    model = models.__dict__[cfg['MODEL']['ARCH']](pretrained=True)
    model = model.to(DEVICE)
    print('ARCH : {}'.format(cfg['MODEL']['ARCH']))
    if use_center_bias_layer:
        print('USE CENTER BIAS LAYER')
        center_bias_layer = CenterBiasLayer().to(DEVICE)
    if use_equator_bias_layer:
        print('USE EQUATOR BIAS LAYER')
        print('EQUATOR BIAS CHANNELS : {}'.format(train_dataset.num_vertical))
        equator_bias_layer = EquatorBiasLayer(train_dataset.num_vertical).to(DEVICE)
    if use_constant_equator_bias:
        print('USE CONSTANT EQUATOR BIAS')
        # TODO from cfg file
        constant_equator_bias_path = os.path.splitext(cfg['DATA']['DATASET_CFG_PATH'])[
                                         0] + '_extracted_constant_bias.npy'
        equator_bias_layer = EquatorBiasLayer(train_dataset.num_vertical,
                                              constant_bias_path=constant_equator_bias_path).to(DEVICE)
    if use_fusion_layer:
        print('USE FUSION LAYER')
        fusion_layer = FusionLayer().to(DEVICE)
    if use_attention_layer:
        if use_basic_attention and not use_basic_attention_with_features:
            print('USE BASIC ATTENTION LAYER')
            attention_layer = Attention(in_ch=len(cfg['MODEL']['VIEW_ANGLE']),
                                        ch=int(cfg['SETTING']['BIAS']['ATTENTION_LAYER_CH'])).to(DEVICE)
        elif use_basic_attention_v2 and not use_basic_attention_with_features:
            print('USE BOTTLENECK ATTENTION LAYER V2')
            attention_layer = AttentionV2(in_ch=len(cfg['MODEL']['VIEW_ANGLE']),
                                          ch=int(cfg['SETTING']['BIAS']['ATTENTION_LAYER_CH'])).to(DEVICE)
        elif use_basic_attention and use_basic_attention_with_features:
            print('USE BOTTLENECK ATTENTION LAYER With Features')
            if cfg['MODEL']['ARCH'] == 'densesalbi3':
                features_ch = 4416
            elif cfg['MODEL']['ARCH'] == 'dpnsal131_dilation_multipath':
                features_ch = 5376
            elif cfg['MODEL']['ARCH'] == 'deepgaze2e':
                features_ch = 4
            attention_layer = AttentionWithFeatures(in_ch=len(cfg['MODEL']['VIEW_ANGLE']),
                                          ch=int(cfg['SETTING']['BIAS']['ATTENTION_LAYER_CH']), 
                                          features_ch=features_ch ).to(DEVICE)
        elif use_basic_attention_v2 and use_basic_attention_with_features:
            print('USE BOTTLENECK ATTENTION LAYER V2 With Features')
            if cfg['MODEL']['ARCH'] == 'densesalbi3':
                features_ch = 4416
            elif cfg['MODEL']['ARCH'] == 'dpnsal131_dilation_multipath':
                features_ch = 5376
            elif cfg['MODEL']['ARCH'] == 'deepgaze2e':
                features_ch = 4
            attention_layer = AttentionV2WithFeatures(in_ch=len(cfg['MODEL']['VIEW_ANGLE']),
                                                    ch=int(cfg['SETTING']['BIAS']['ATTENTION_LAYER_CH']), 
                                                    features_ch=features_ch ).to(DEVICE)

    print("hoge")
    # load checkpoint
    start_epoch = 0
    best_val_loss = 1e5
    # Loading OSIE pretrained model as default
    if cfg['SETTING']['RESUME'] is not None:
        # Loading paramteres of OSIE pretrained model
        resume_path = os.path.abspath(cfg['SETTING']['RESUME'])
        if not (os.path.isfile(resume_path)):
            raise OSError('No checkpoint file found at {}'.format(resume_path))
        print('LOADING CHECKPOINT {}'.format(resume_path))
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        if 'epoch' in checkpoint.keys():
            print('LOADED (epoch {})'.format(checkpoint['epoch']))
        else:
            print('LOADED (epoch is unknown)'
                  .format(resume_path))
        if use_center_bias_layer:
            if 'center_bias_state_dict' in checkpoint:
                print("LOADING CENTER BIAS...")
                center_bias_layer.load_state_dict(checkpoint['center_bias_state_dict'])
                print("LOADED")
        if use_equator_bias_layer:
            if (cfg['SETTING']['BIAS']['EQUATOR_BIAS_LAYER_LOADING_CENTER_BIAS']):
                print("EQUATOR BIAS LAYER LOADING CENTER BIAS...")
                if 'center_bias_state_dict' in checkpoint:
                    equator_bias_layer.load_center_bias(checkpoint)
                    print("LOADED")
                else:
                    print("CENTER BIAS DO NOT INCLUDE IN THIS CHECKPOINT")
                    print("EQUATOR BIAS LAYER LOADING EQUATOR BIAS...")
                    equator_bias_layer.load_state_dict(checkpoint['equator_bias_state_dict'])
                    print("LOADED")
            elif 'equator_bias_state_dict' in checkpoint:
                print("LOADING EQUATOR BIAS...")
                equator_bias_layer.load_state_dict(checkpoint['equator_bias_state_dict'])
                print("LOADED")
    else:
        print('using the model pretrained on ImageNet')

    parameters = []
    if not (cfg['SETTING']['MAIN']['FIX_MAIN_NET']):
        parameters.append({'params': model.parameters()})
    else:
        print('fix main net param')
    if use_center_bias_layer:
        parameters.append({'params': center_bias_layer.parameters(), 'lr': float(cfg['SETTING']['BIAS']['LR'])})
    else:
        center_bias_layer = None
    if use_equator_bias_layer:
        parameters.append({'params': equator_bias_layer.parameters(), 'lr': float(cfg['SETTING']['BIAS']['LR'])})
    else:
        if use_constant_equator_bias:
            pass
        else:
            equator_bias_layer = None

    if use_fusion_layer:
        parameters.append({'params': fusion_layer.parameters(), 'lr': float(cfg['SETTING']['BIAS']['LR'])})
    else:
        fusion_layer = None

    if use_attention_layer:
        parameters.append({'params': attention_layer.parameters(), 'lr': float(cfg['SETTING']['BIAS']['LR'])})
    else:
        attention_layer = None

    try:
        assert len(parameters) != 0, 'No parameters to train.'
    except AssertionError as err:
        print('AssertionError : {}'.format(err))

    # define loss function (criterion) and optimizer
    # RMSprop or SGD
    # https://rightcode.co.jp/blog/information-technology/torch-optim-optimizer-compare-and-verify-update-process-and-performance-of-optimization-methods
    criterion = KLD().to(DEVICE)
    if cfg['SETTING']['OPTIMIZER'] == 'RMSprop':
        optimizer = torch.optim.RMSprop(parameters, float(cfg['SETTING']['MAIN']['LR']),
                                        alpha=float(cfg['SETTING']['ALPHA']),
                                        weight_decay=float(cfg['SETTING']['WEIGHT_DECAY']))
    elif cfg['SETTING']['OPTIMIZER'] == 'sgd':
        optimizer = torch.optim.SGD(parameters, float(cfg['SETTING']['MAIN']['LR']), momentum=0.9,
                                    weight_decay=float(cfg['SETTING']['WEIGHT_DECAY']))

    save_ckpt_dir = os.path.join(cfg['DIR']['ROOT_LOG_DIR'], train_dataset.dataset_name, cfg['MODEL']['ARCH'], 'ckpt')
    if not (os.path.isdir(save_ckpt_dir)):
        os.makedirs(save_ckpt_dir)
    print('-----------------------')

    # train

    epoch = -1
    time_stamp = '{0:%Y%m%d-%H%M%S}'.format(datetime.now())
    train_loss = train(epoch, optimizer, model, center_bias_layer, equator_bias_layer, fusion_layer, attention_layer,
                       train_dataset, train_loader, criterion, use_center_bias_layer=use_center_bias_layer,
                       use_equator_bias_layer=use_equator_bias_layer, use_fusion_layer=use_fusion_layer,
                       use_attention_layer=use_attention_layer, use_attention_features=use_basic_attention_with_features,
                       arch=cfg['MODEL']['ARCH']
                       )
    if has_val:
        val_loss = val(epoch, model, center_bias_layer, equator_bias_layer, fusion_layer, attention_layer, val_dataset,
                       val_loader, criterion, use_center_bias_layer=use_center_bias_layer,
                       use_equator_bias_layer=use_equator_bias_layer, use_fusion_layer=use_fusion_layer,
                       use_attention_layer=use_attention_layer, use_attention_features=use_basic_attention_with_features,
                       arch=cfg['MODEL']['ARCH']
                       )
    else:
        val_loss = 0
    print('train loss before training : {}'.format(train_loss))
    if has_val:
        print('val loss before training : {}'.format(val_loss))
    if use_center_bias_layer:
        center_bias_weight = center_bias_layer.weight.data[0, 0, :, :].cpu().numpy()
        scaled_center_bias_weight = (255 * (center_bias_weight - center_bias_weight.min()) / (
                center_bias_weight.max() - center_bias_weight.min())).astype(np.uint8)
        out_weight_file_png = os.path.join(save_ckpt_dir, 'center_bias_{}_epoch{}.png'.format(start_time_stamp, epoch))
        out_weight_file_npy = os.path.join(save_ckpt_dir, 'center_bias_{}_epoch{}.npy'.format(start_time_stamp, epoch))
        imageio.imwrite(out_weight_file_png, scaled_center_bias_weight)
        np.save(out_weight_file_npy, center_bias_weight)
        bias_max = center_bias_weight.max()
        bias_min = center_bias_weight.min()
    elif use_equator_bias_layer:
        bias_max = 0
        bias_min = 1e5
        for i in range(train_dataset.num_vertical):
            equator_bias_weight = equator_bias_layer.weights[i].data[0, 0, :, :].cpu().numpy()
            if equator_bias_weight.max() == equator_bias_weight.min():
                scaled_equator_bias_weight = (255 * np.ones(equator_bias_weight.shape)).astype(np.uint8)
            else:
                scaled_equator_bias_weight = (255 * (equator_bias_weight - equator_bias_weight.min()) / (
                        equator_bias_weight.max() - equator_bias_weight.min())).astype(np.uint8)
            out_weight_file_png = os.path.join(save_ckpt_dir,
                                               'equator_bias_{}_epoch{}_{}.png'.format(start_time_stamp, epoch, i))
            out_weight_file_npy = os.path.join(save_ckpt_dir,
                                               'equator_bias_{}_epoch{}_{}.npy'.format(start_time_stamp, epoch, i))
            imageio.imwrite(out_weight_file_png, scaled_equator_bias_weight)
            np.save(out_weight_file_npy, equator_bias_weight)
            bias_max = max(bias_max, equator_bias_weight.max())
            bias_min = min(bias_min, equator_bias_weight.min())
    else:
        bias_max = 0
        bias_min = 0

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'train_loss', 'val_loss', 'is_best', 'time_stamp', 'bias_max', 'bias_min'
    ])
    log_tmp = pd.Series([epoch, train_loss, val_loss, False, time_stamp, bias_max, bias_min],
                        index=['epoch', 'train_loss', 'val_loss', 'is_best', 'time_stamp', 'bias_max', 'bias_min'])
    log = log.append(log_tmp, ignore_index=True)

    for epoch in range(start_epoch, cfg['SETTING']['EPOCHS']):
        time_stamp = '{0:%Y%m%d-%H%M%S}'.format(datetime.now())
        train_loss = train(epoch, optimizer, model, center_bias_layer, equator_bias_layer, fusion_layer,
                           attention_layer, train_dataset, train_loader, criterion,
                           use_center_bias_layer=use_center_bias_layer, use_equator_bias_layer=use_equator_bias_layer,
                           use_fusion_layer=use_fusion_layer, use_attention_layer=use_attention_layer,
                           use_attention_features=use_basic_attention_with_features,
                           arch=cfg['MODEL']['ARCH']
                           )
        if has_val:
            val_loss = val(epoch, model, center_bias_layer, equator_bias_layer, fusion_layer, attention_layer,
                           val_dataset, val_loader, criterion, use_center_bias_layer=use_center_bias_layer,
                           use_equator_bias_layer=use_equator_bias_layer, use_fusion_layer=use_fusion_layer,
                           use_attention_layer=use_attention_layer, use_attention_features=use_basic_attention_with_features,
                           arch=cfg['MODEL']['ARCH']
                           )
            # check best epoch
            is_best = val_loss < best_val_loss
            best_val_loss = min(val_loss, best_val_loss)
        else:
            val_loss = 0
            is_best = False
            best_val_loss = 0

        save_dict = {
            'epoch': epoch,
            'cfg': cfg,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'mean_rgb': train_dataset.mean_rgb,
            'is_best': is_best,
            'best_val_loss': best_val_loss,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'start_time_stamp': start_time_stamp,
            'time_stamp': time_stamp
        }

        if use_center_bias_layer:
            save_dict['center_bias_state_dict'] = center_bias_layer.state_dict()
            center_bias_weight = center_bias_layer.weight.data[0, 0, :, :].cpu().numpy()
            scaled_center_bias_weight = (255 * (center_bias_weight - center_bias_weight.min()) / (
                    center_bias_weight.max() - center_bias_weight.min())).astype(np.uint8)
            out_weight_file_png = os.path.join(save_ckpt_dir,
                                               'center_bias_{}_epoch{}.png'.format(start_time_stamp, epoch))
            out_weight_file_npy = os.path.join(save_ckpt_dir,
                                               'center_bias_{}_epoch{}.npy'.format(start_time_stamp, epoch))
            imageio.imwrite(out_weight_file_png, scaled_center_bias_weight)
            np.save(out_weight_file_npy, center_bias_weight)
            bias_max = center_bias_weight.max()
            bias_min = center_bias_weight.min()
            if is_best:
                best_cb_max = center_bias_weight.max()
                best_cb_min = center_bias_weight.min()
        if use_equator_bias_layer:
            save_dict['equator_bias_state_dict'] = equator_bias_layer.state_dict()
            save_dict['equator_bias_channels'] = train_dataset.num_vertical
            bias_max = 0
            bias_min = 1e5
            for i in range(train_dataset.num_vertical):
                equator_bias_weight = equator_bias_layer.weights[i].data[0, 0, :, :].cpu().numpy()
                if equator_bias_weight.max() == equator_bias_weight.min():
                    scaled_equator_bias_weight = (255 * np.ones(equator_bias_weight.shape)).astype(np.uint8)
                else:
                    scaled_equator_bias_weight = (255 * (equator_bias_weight - equator_bias_weight.min()) / (
                            equator_bias_weight.max() - equator_bias_weight.min())).astype(np.uint8)
                out_weight_file_png = os.path.join(save_ckpt_dir,
                                                   'equator_bias_{}_epoch{}_{}.png'.format(start_time_stamp, epoch, i))
                out_weight_file_npy = os.path.join(save_ckpt_dir,
                                                   'equator_bias_{}_epoch{}_{}.npy'.format(start_time_stamp, epoch, i))
                imageio.imwrite(out_weight_file_png, scaled_equator_bias_weight)
                np.save(out_weight_file_npy, equator_bias_weight)
                bias_max = max(bias_max, equator_bias_weight.max())
                bias_min = min(bias_min, equator_bias_weight.min())
        if use_constant_equator_bias:
            save_dict['use_constant_equator_bias'] = True

        if use_fusion_layer:
            save_dict['fusion_layer_state_dict'] = fusion_layer.state_dict()
        if use_attention_layer:
            save_dict['attention_layer_state_dict'] = attention_layer.state_dict()

        log_tmp = pd.Series([epoch, train_loss, val_loss, is_best, time_stamp, bias_max, bias_min],
                            index=['epoch', 'train_loss', 'val_loss', 'is_best', 'time_stamp', 'bias_max', 'bias_min'])
        log = log.append(log_tmp, ignore_index=True)

        log_path = os.path.join(save_ckpt_dir, 'log_{}.csv'.format(start_time_stamp))
        log.to_csv(log_path, index=False)
        save_checkpoint(save_dict, is_best, save_dir=save_ckpt_dir, save_step=cfg['SETTING']['SAVE_STEP'])
        print('Save Log Path : {}'.format(save_ckpt_dir))

    print('--------------------------------')
    print('DONE')
    print('START TIME : {}'.format(start_time_stamp))
    time = datetime.now()
    finish_time_stamp = '{0:%Y%m%d-%H%M%S}'.format(time)
    print('FINISH TIME : {}'.format(finish_time_stamp))
    checkpoint_path = os.path.join(save_ckpt_dir,
                                   'checkpoint_{}_epoch{}.pth.tar'.format(start_time_stamp, save_dict['epoch']))
    print('CHECKPOINT PATH : {}'.format(checkpoint_path))
    if has_val:
        best_checkpoint_path = os.path.join(save_ckpt_dir, 'checkpoint_{}_model_best.pth.tar'.format(start_time_stamp))
        print('BEST CHECKPOINT PATH : {}'.format(best_checkpoint_path))
    print('--------------------------------')

    if has_val:
        return best_checkpoint_path, log_path
    else:
        return checkpoint_path, log_path


if __name__ == '__main__':
    args = get_parser().parse_args()
    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f)
    main(cfg['TRAIN'])
