import models
from models.readout_nets.center_bias_layer import CenterBiasLayer
from models.readout_nets.equator_bias_layer import EquatorBiasLayer
import odimapping
from utils.loaders import TestPlanarLoader, MultiScaleTestODILoader
from utils import np_transforms
from utils.dataset import TestPlanarDataset, MultiTestODIDataset

import argparse
from datetime import datetime
import glob
import os

from models.fusion_layer import FusionLayer
from models.attention_layer import Attention
from models.attention_layer import AttentionV2
from models.attention_layer import AttentionWithFeatures
from models.attention_layer import AttentionV2WithFeatures
from models.composite import composite_avg
from models.composite import composite_max
from odimapping.unify import cropping

import cv2
import imageio
import numpy as np
import pickle
from PIL import Image
import torch
from tqdm import tqdm
import yaml

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

Image.MAX_IMAGE_PIXELS = 1000000000


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/sample.yaml',
                        help='config file')
    return parser


def main(cfg):
    start_time_stamp = '{0:%Y%m%d-%H%M%S}'.format(datetime.now())
    print('-----------------------')
    print('START TIME : {}'.format(start_time_stamp))

    use_equator_bias_layer = cfg['SETTING']['BIAS']['USE_EQUATOR_BIAS_LAYER']
    use_center_bias_layer = cfg['SETTING']['BIAS']['USE_CENTER_BIAS_LAYER']
    use_constant_equator_bias = cfg['SETTING']['BIAS']['USE_CONSTANT_EQUATOR_BIAS']
    use_fusion_layer = cfg['SETTING']['BIAS']['USE_FUSION_LAYER']
    use_attention_layer = cfg['SETTING']['BIAS']['USE_ATTENTION_LAYER']
    use_basic_attention = cfg['SETTING']['BIAS']['USE_BASIC_ATTENTION_LAYER']
    use_basic_attention_v2 = cfg['SETTING']['BIAS']['USE_BASIC_ATTENTION_V2_LAYER']
    use_basic_attention_with_features = cfg['SETTING']['BIAS']['USE_BASIC_ATTENTION_WITH_FEATURE']
    use_composite_max = False
    if 'USE_COMPOSITE_MAX' in cfg.keys():
        use_composite_max = cfg['SETTING']['BIAS']['USE_COMPOSITE_MAX']
    use_composite_avg = True
    if 'USE_COMPOSITE_AVG' in cfg.keys():
        use_composite_max = cfg['SETTING']['BIAS']['USE_COMPOSITE_AVG']

    batch_size = cfg['SETTING']['BATCH_SIZE']
    image_files = glob.glob(cfg["DATA"]["IMAGE_FILES"])

    model_names = sorted(name for name in models.__dict__
                         if not name.startswith("_") and callable(models.__dict__[name]))

    view_angles = cfg['MODEL']['VIEW_ANGLE']

    out_dir = os.path.join(cfg['DIR']['ROOT_OUTPUT_DIR'], cfg['MODEL']['ARCH'], start_time_stamp)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print("Estimated result dir path is " + out_dir)

    if cfg['MODEL']['EXTRACT'] is not None:
        '''
        create mapping object for predicting saliency-map for odi
        '''
        mappings = []
        for view_angle in view_angles:
            if cfg['DATA']['PROJECTION'] == 'equirectangular':
                mappings.append(odimapping.equirectangular(
                    cfg['MODEL']['EXTRACT'], cfg['DIR']['MAPS_DIR'],
                    view_angle_p=view_angle, view_angle_t=view_angle,
                    extract_h=cfg['DATA']['EXTRACT_HEIGHT'],
                    extract_w=cfg['DATA']['EXTRACT_WIDTH'],
                    odi_h=cfg['DATA']['RESIZE_HEIGHT'],
                    odi_w=cfg['DATA']['RESIZE_WIDTH']))
            else:
                raise RuntimeError(
                    'Expected projection is not planar. Now, projection is {}.'.format(cfg['DATA']['PROJECTION']))

        print('EXTRACTION METHOD : {}'.format(cfg['MODEL']['EXTRACT']))

    # load model
    model = models.__dict__[cfg['MODEL']['ARCH']](pretrained=False).to(DEVICE)

    if use_center_bias_layer:
        print('use center bias layer')
        center_bias_layer = CenterBiasLayer().to(DEVICE)
    if use_equator_bias_layer:
        print('use equator bias layer')
        equator_bias_layer = EquatorBiasLayer(mappings[0].num_vertical).to(DEVICE)

    if use_fusion_layer:
        print('use fusion layer')
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

    # load checkpoint
    print(os.path.isfile(cfg['SETTING']['RESUME']))
    if os.path.isfile(cfg['SETTING']['RESUME']):
        print('LOADING CHECKPOINT {}'.format(cfg['SETTING']['RESUME']))
        checkpoint = torch.load(cfg['SETTING']['RESUME'])
        model.load_state_dict(checkpoint['state_dict'], strict=True)

        mean_rgb = checkpoint['mean_rgb']
        print("LOADED")

        if use_center_bias_layer:
            if 'center_bias_state_dict' in checkpoint:
                print("LOADING CENTER BIAS...")
                center_bias_layer.load_state_dict(checkpoint['center_bias_state_dict'])
                print("LOADED")
                # save image of center_bias
                center_bias_weight = center_bias_layer.weight.data[0, 0, :, :].cpu().numpy()
                scaled_center_bias_weight = (255 * (center_bias_weight - center_bias_weight.min()) / (
                            center_bias_weight.max() - center_bias_weight.min())).astype(np.uint8)
                out_weight_file_png = os.path.join(out_dir, 'center_bias.png')
                out_weight_file_npy = os.path.join(out_dir, 'center_bias.npy')
                imageio.imwrite(out_weight_file_png, scaled_center_bias_weight)
                np.save(out_weight_file_npy, scaled_center_bias_weight)
        if use_equator_bias_layer:
            if 'equator_bias_state_dict' in checkpoint:
                print("LOADING EQUATOR BIAS...")
                equator_bias_layer.load_state_dict(checkpoint['equator_bias_state_dict'])
                print("LOADED")
                for i in range(checkpoint['equator_bias_channels']):
                    equator_bias_weight = equator_bias_layer.weights[i].data[0, 0, :, :].cpu().numpy()
                    if equator_bias_weight.max() == equator_bias_weight.min():
                        scaled_equator_bias_weight = (255 * np.ones(equator_bias_weight.shape)).astype(np.uint8)
                    else:
                        scaled_equator_bias_weight = (255 * (equator_bias_weight - equator_bias_weight.min()) / (
                                    equator_bias_weight.max() - equator_bias_weight.min())).astype(np.uint8)
                    out_weight_file_png = os.path.join(out_dir, 'equator_bias_{}.png'.format(i))
                    out_weight_file_npy = os.path.join(out_dir, 'equator_bias_{}.npy'.format(i))
                    imageio.imwrite(out_weight_file_png, scaled_equator_bias_weight)
                    np.save(out_weight_file_npy, equator_bias_weight)
            else:
                raise RuntimeError(
                    'The weights of equator bias layer do not include in this checkpoint. Please use other checkpoint.')
        if use_fusion_layer:
            if 'fusion_layer_state_dict' in checkpoint:
                print("LOADING FUSION LAYER STATE...")
                fusion_layer.load_state_dict(checkpoint['fusion_layer_state_dict'])
                print("LOADED")
            else:
                raise RuntimeError(
                    'The weights of fusion layer do not include in this checkpoint. Please use other checkpoint.')
        if use_attention_layer:
            if 'attention_layer_state_dict' in checkpoint:
                print("LOADING ATTENTION LAYER STATE...")
                attention_layer.load_state_dict(checkpoint['attention_layer_state_dict'])
                print("LOADED")
            else:
                raise RuntimeError(
                    'The weights of attention layer do not include in this checkpoint. Please use other checkpoint.')

    else:
        raise OSError('No checkpoint file found at {}'.format(cfg['SETTING']['RESUME']))

    # create loader
    all_extracted_files = [[] for i in range(mappings[0].num_extracted * len(image_files))]
    print('create loader...')
    if cfg['MODEL']['EXTRACT'] is not None:
        '''
        create loader for predicting saliency-map for odi
        '''
        for i in range(len(mappings)):
            extracted_files, eqbl_idxs, extraction_idxs = mappings[i].load_extract_save_(image_files, view_angles[i])
            for j in range(len(all_extracted_files)):
                all_extracted_files[j].append(extracted_files[j])

        test_dataset = MultiTestODIDataset(
            all_extracted_files, mean_rgb, eqbl_idxs, extraction_idxs,
            transform=np_transforms.Compose([
                np_transforms.ToTensor()
            ])
        )
        test_loader = MultiScaleTestODILoader(test_dataset, batch_size=batch_size)
        print('Done')

    else:
        '''
        create loader for predict saliency-map for planar iamge
        '''
        test_dataset = TestPlanarDataset(
            image_files, mean_rgb,
            size=(cfg['DATA']['RESIZE_WIDTH'], cfg['DATA']['RESIZE_HEIGHT']),
            transform=np_transforms.Compose([
                np_transforms.ToTensor(),
            ])
        )
        print('Done')

    # estimate
    with torch.no_grad():
        if cfg['MODEL']['EXTRACT'] is not None:
            save_files_all = []
            extraction_idxs_all = []
            save_dir = os.path.join(out_dir, 'extracted_im_saliencymap')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        else:
            save_dir = os.path.join(out_dir, 'saliencymap')
            print('save_dir : {}'.format(save_dir))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        # if use_basic_attention_with_features and cfg['MODEL']['ARCH'] != 'deepgaze2e':
        # # features are returned from model.forward for 'deepgaze2e'
            
        #     def forward_hook(module, inputs, outputs):
        #         global features
        #         # norm5.outだと何かおかしい
        #         features = inputs[0]

        #     # densenet161の最終出力特徴量を取得
        #     if cfg['MODEL']['ARCH'] == 'dpnsal131_dilation_multipath':
        #         target_module = model.readout_net
        #     else:
        #         target_module = model.readout_net[0]
        #     handle = target_module.register_forward_hook(forward_hook)
            
        for i, data in tqdm(enumerate(test_loader, start=1)):
            min_angle_features = None
            all_outputs = []
            file_names = data[0]
            inputs = data[1]
            if cfg['MODEL']['EXTRACT'] is not None:
                eqbl_idxs = data[2]
                extraction_idxs = data[3]

            for j in range(len(inputs)):
                inputs[j] = inputs[j].to(DEVICE)  # inputs[j]: b, c, h, w
                input_image = inputs[j]
                if use_basic_attention_with_features:
                    if cfg['MODEL']['ARCH'] == 'deepgaze2e':
                        centerbias_tensor = torch.tensor(np.zeros((1, input_image.shape[2], input_image.shape[3]))).to(DEVICE)
                        [log_density, features] = model(input_image, centerbias_tensor)
                        outputs = (torch.exp(log_density)).float()
                        features = features.float()
                        features = features.detach() # do not feedback grad. https://qiita.com/ground0state/items/15f218ab89121d66b462
                    else:
                        [outputs, features] = model(input_image)
                        #outputs = model(inputs[j])  # outputs: b, c, h, w
                else:
                    outputs = model(input_image)

                if use_center_bias_layer:
                    outputs = center_bias_layer(outputs)
                if use_equator_bias_layer:
                    for num in range(inputs[j].size(0)):
                        outputs[num] = equator_bias_layer(outputs[num].unsqueeze(0), eqbl_idxs[num])[0]
                if j != 0:
                    outputs = cropping(view_angles[0], view_angles[j], outputs)
                if j == 0 and use_basic_attention_with_features:
                    min_angle_features = features

                all_outputs.append(outputs)

            all_outputs = torch.cat(all_outputs, dim=1)
            if use_fusion_layer:
                outputs = fusion_layer(all_outputs)
            if use_attention_layer:
                if use_basic_attention_with_features:
                    outputs = attention_layer(all_outputs, min_angle_features)
                else:
                    outputs = attention_layer(all_outputs)

            if cfg['MODEL']['EXTRACT'] is not None:
                if (mappings[0].num_extracted - 1) in extraction_idxs:
                    index = extraction_idxs.index(mappings[0].num_extracted - 1)
                    outputs_tmp = torch.cat((outputs_tmp, outputs[:index + 1]))
                    file_names_tmp.extend(file_names[:index + 1])
                    save_files = mappings[0].save_before_embedding(outputs_tmp, save_dir, file_names_tmp)
                    save_files_all.extend(save_files)
                    extraction_idxs_all.extend(extraction_idxs)
                    if index != outputs.size(0):
                        outputs_tmp = outputs[index + 1:]
                        file_names_tmp = file_names[index + 1:]
                else:
                    extraction_idxs_all.extend(extraction_idxs)
                    if i == 1:
                        outputs_tmp = outputs
                        file_names_tmp = file_names
                    else:
                        outputs_tmp = torch.cat((outputs_tmp, outputs))
                        file_names_tmp.extend(file_names)
            else:
                '''
                save planar saliency-map
                '''
                for num in range(inputs.size(0)):
                    saliencymap = outputs.data[num].cpu().numpy()
                    saliencymap = np.uint8(255 * saliencymap / saliencymap.max())
                    if (len(saliencymap.shape) == 3) and (saliencymap.shape[0] == 1):
                        saliencymap = saliencymap[0]
                    file_name = file_names[num] + '.png'
                    save_path = os.path.join(save_dir, file_name)
                    # print('-- file_name :',file_name)
                    imageio.imwrite(save_path, saliencymap)

        if cfg['MODEL']['EXTRACT'] is not None:
            '''
            save odi saliency-map
            '''
            save_dir = os.path.join(out_dir, 'ODIsaliencymap')
            print('save_dir : {}'.format(save_dir))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            p0 = 0
            for i in tqdm(range(len(save_files_all))):
                if extraction_idxs_all[i] == (mappings[0].num_extracted - 1):
                    if use_constant_equator_bias:
                        odi_salmap = mappings[0].load_embed(save_files_all[p0:i + 1],
                                                            use_constant_equator_bias=True,
                                                            constant_equator_bias_path=cfg["DATA"][
                                                                "CONSTANT_EQUATOR_BIAS_PATH"])
                    else:
                        odi_salmap = mappings[0].load_embed(save_files_all[p0:i + 1])
                    file_name = ('_').join(os.path.basename(save_files_all[p0:i + 1][0]).split('_')[:-1]) + '.png'
                    save_path = os.path.join(save_dir, file_name)
                    imageio.imwrite(save_path, odi_salmap)
                    color_salmap = cv2.applyColorMap((odi_salmap).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
                    color_file_name = file_name = ('_').join(
                        os.path.basename(save_files_all[p0:i + 1][0]).split('_')[:-1]) + '_color.png'
                    color_save_path = os.path.join(save_dir, color_file_name)
                    imageio.imwrite(color_save_path, color_salmap)
                    p0 = i + 1
            save_pickle = os.path.join(save_dir, 'cfg.pickle')
            with open(save_pickle, mode='wb') as f:
                pickle.dump(cfg, f)

    return save_dir


if __name__ == '__main__':
    args = get_parser().parse_args()
    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f)
    main(cfg['ESTIMATE'])
