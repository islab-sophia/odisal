import argparse
from datetime import datetime
import os
import struct

import numpy as np
from numpy import random
import pandas as pd
from PIL import Image
from scipy.ndimage import filters
from skimage.transform import resize
from tqdm import tqdm
import yaml

#import UtilityTools.UtilityTools as tools

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/sample.yaml',
                        help='config file')
    return parser

def normalize(x, method='standard', axis=None):
    x = np.array(x, copy=False)
    if axis is not None:
        y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
        shape = np.ones(len(x.shape))
        shape[axis] = x.shape[axis]
        if method == 'standard':
            res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
        elif method == 'range':
            res = (x - np.min(y, axis=1).reshape(shape)) / (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
        elif method == 'sum':
            res = x / np.float_(np.sum(y, axis=1).reshape(shape))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    else:
        if method == 'standard':
            res = (x - np.mean(x)) / np.std(x)
        elif method == 'range':
            res = (x - np.min(x)) / (np.max(x) - np.min(x))
        elif method == 'sum':
            res = x / float(np.sum(x))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    return res

def KLD(salMap, posMap):
    eps = np.finfo(np.float64).eps
    fm = posMap / (np.sum(posMap) + eps)
    sm = salMap / (np.sum(salMap) + eps)
    return np.sum( fm * np.log( (fm + eps) / (sm + eps) ) )

def AUC_Judd(saliency_map, fixation_map, jitter=False):
    saliency_map = np.array(saliency_map, copy=False)
    fixation_map = np.array(fixation_map, copy=False) > 0.5
    # If there are no fixation to predict, return NaN
    if not np.any(fixation_map):
        print('no fixation to predict')
        return np.nan
    # Make the saliency_map the size of the fixation_map
    if saliency_map.shape != fixation_map.shape:
        saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='constant')
    # Jitter the saliency map slightly to disrupt ties of the same saliency value
    if jitter:
        saliency_map += random.rand(*saliency_map.shape) * 1e-7
    # Normalize saliency map to have values between [0,1]
    saliency_map = normalize(saliency_map, method='range')

    S = saliency_map.ravel()
    F = fixation_map.ravel()
    S_fix = S[F] # Saliency map values at fixation locations
    n_fix = len(S_fix)
    n_pixels = len(S)
    # Calculate AUC
    thresholds = sorted(S_fix, reverse=True)
    tp = np.zeros(len(thresholds)+2)
    fp = np.zeros(len(thresholds)+2)
    tp[0] = 0; tp[-1] = 1
    fp[0] = 0; fp[-1] = 1
    for k, thresh in enumerate(thresholds):
        above_th = np.sum(S >= thresh) # Total number of saliency map values above threshold
        tp[k+1] = (k + 1) / float(n_fix) # Ratio saliency map values at fixation locations above threshold
        fp[k+1] = (above_th - k - 1) / float(n_pixels - n_fix) # Ratio other saliency map values above threshold
    return np.trapz(tp, fp) # y, x

def AUC_Borji(saliency_map, fixation_map, n_rep=100, step_size=0.1, rand_sampler=None):
    saliency_map = np.array(saliency_map, copy=False)
    fixation_map = np.array(fixation_map, copy=False) > 0.5
    # If there are no fixation to predict, return NaN
    if not np.any(fixation_map):
        print('no fixation to predict')
        return np.nan
    # Make the saliency_map the size of the fixation_map
    if saliency_map.shape != fixation_map.shape:
        saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='constant')
    # Normalize saliency map to have values between [0,1]
    saliency_map = normalize(saliency_map, method='range')

    S = saliency_map.ravel()
    F = fixation_map.ravel()
    S_fix = S[F] # Saliency map values at fixation locations
    n_fix = len(S_fix)
    n_pixels = len(S)
    # For each fixation, sample n_rep values from anywhere on the saliency map
    if rand_sampler is None:
        r = random.randint(0, n_pixels, [n_fix, n_rep])
        S_rand = S[r] # Saliency map values at random locations (including fixated locations!? underestimated)
    else:
        S_rand = rand_sampler(S, F, n_rep, n_fix)
    # Calculate AUC per random split (set of random locations)
    auc = np.zeros(n_rep) * np.nan
    for rep in range(n_rep):
        thresholds = np.r_[0:np.max(np.r_[S_fix, S_rand[:,rep]]):step_size][::-1]
        tp = np.zeros(len(thresholds)+2)
        fp = np.zeros(len(thresholds)+2)
        tp[0] = 0; tp[-1] = 1
        fp[0] = 0; fp[-1] = 1
        for k, thresh in enumerate(thresholds):
            tp[k+1] = np.sum(S_fix >= thresh) / float(n_fix)
            fp[k+1] = np.sum(S_rand[:,rep] >= thresh) / float(n_fix)
        auc[rep] = np.trapz(tp, fp)
    return np.mean(auc) # Average across random splits

def NSS(saliency_map, fixation_map):
    s_map = np.array(saliency_map, copy=False)
    f_map = np.array(fixation_map, copy=False) > 0.5
    if s_map.shape != f_map.shape:
        s_map = resize(s_map, f_map.shape)
    # Normalize saliency map to have zero mean and unit std
    s_map = normalize(s_map, method='standard')
    # Mean saliency value at fixation locations
    return np.mean(s_map[f_map])


def CC(saliency_map1, saliency_map2):
    map1 = np.array(saliency_map1, copy=False)
    map2 = np.array(saliency_map2, copy=False)
    if map1.shape != map2.shape:
        map1 = resize(map1, map2.shape, order=3, mode='constant') # bi-cubic/nearest is what Matlab imresize() does by default
    # Normalize the two maps to have zero mean and unit std
    map1 = normalize(map1, method='standard')
    map2 = normalize(map2, method='standard')
    # Compute correlation coefficient
    return np.corrcoef(map1.ravel(), map2.ravel())[0,1]


def SIM(saliency_map1, saliency_map2):
    map1 = np.array(saliency_map1, copy=False)
    map2 = np.array(saliency_map2, copy=False)
    if map1.shape != map2.shape:
        map1 = resize(map1, map2.shape, order=3, mode='constant') # bi-cubic/nearest is what Matlab imresize() does by default
    # Normalize the two maps to have values between [0,1] and sum up to 1
    map1 = normalize(map1, method='range')
    map2 = normalize(map2, method='range')
    map1 = normalize(map1, method='sum')
    map2 = normalize(map2, method='sum')
    # Compute histogram intersection
    intersection = np.minimum(map1, map2)
    return np.sum(intersection)
#### METRICS --

def getSimVal(salmap_pred, salmap_groundtruth, fixmap_groundtruth, keys_order):
    values = []
    for metric in keys_order:
        func = metrics[metric][0]
        compType = metrics[metric][2]
        if compType == "fix":
            m = func(salmap_pred, fixmap_groundtruth)
        else:
            m = func(salmap_pred, salmap_groundtruth)
        values.append(m)
    return values

def uniformSphereSampling(N):
    gr = (1 + np.sqrt(5))/2
    ga = 2 * np.pi * (1 - 1/gr)

    ix = iy = np.arange(N)

    lat = np.arccos(1 - 2*ix/(N-1))
    lon = iy * ga
    lon %= 2*np.pi

    return np.concatenate([lat[:, None], lon[:, None]], axis=1)

def equi2sinusoidal_fix(fixations, VerticalWeighting, input_width):
    fixMap = [[0] * np.int(input_width * w + 0.5) for w in VerticalWeighting]
    for iFix in range(fixations.shape[0]):
        m = len(fixMap[fixations[iFix, 1]])
        c = np.int(fixations[iFix, 0] * (m - 1) / float(input_width - 1) + 0.5) # index = 0:m-1
        fixMap[fixations[iFix, 1]][c] += 1
    return np.array(sum(fixMap, [])) #flatten
    
def main(cfg, view_angle=None):
    start_time_stamp = '{0:%Y%m%d-%H%M%S}'.format(datetime.now())
    eval_dir = cfg["DIR"]["ROOT_EVALUATE_DIR"]
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    
    if view_angle == None:
        save_path = '{}.csv'.format(os.path.join(eval_dir, "{}_{}".format(cfg["SETTING"]["NAME"], start_time_stamp)))
    else:
        save_path = '{}.csv'.format(os.path.join(eval_dir, "{}_{}_{}".format(cfg["SETTING"]["NAME"], str(view_angle), start_time_stamp)))

    print("----------")
    print("SAVE PATH", save_path)

    eps = np.finfo(np.float64).eps
    keys_order = ['NSS', 'CC', 'SIM', 'KLD', 'AUC_Judd']

    metrics = {
        "AUC_Judd": [AUC_Judd, False, 'fix'], # Binary fixation map
        "AUC_Borji": [AUC_Borji, False, 'fix'], # Binary fixation map
        "NSS": [NSS, False, 'fix'], # Binary fixation map
        "CC": [CC, False, 'sal'], # Saliency map
        "SIM": [SIM, False, 'sal'], # Saliency map
        "KLD": [KLD, False, 'sal'] } # Saliency map

    dtypes = {16: np.float16,
              32: np.float32,
              64: np.float64}

    if type(cfg["SETTING"]["SAMPLING_TYPE"]) != list:
        samplig_type_list = [cfg["SETTING"]["SAMPLING_TYPE"]]
    else:
        samplig_type_list = cfg["SETTING"]["SAMPLING_TYPE"]

    if "Sin" or "Sinusoidal" in samplig_type_list:
        VerticalWeighting = np.sin(np.linspace(0, np.pi, cfg["DATA"]["INPUT_HEIGHT"])) # latitude weighting

    if "Sinusoidal" in samplig_type_list:
        sinusoidalBinaryMap = np.full((cfg["DATA"]["INPUT_HEIGHT"], cfg["DATA"]["INPUT_WIDTH"]), False)
        sinusoidal_pos = [(np.linspace(0, cfg["DATA"]["INPUT_WIDTH"]-1, np.int(cfg["DATA"]["INPUT_WIDTH"] * w + 0.5)) + 0.5).astype(int) for w in VerticalWeighting]
        for r in range(len(VerticalWeighting)):
                sinusoidalBinaryMap[r, sinusoidal_pos[r]] = True

    gf_list = list(map(int, cfg["SETTING"]["GAUSSIAN_FILTER"]))
    filepath_df = pd.read_csv(cfg["DATA"]["EVALUATE_DATASET_CSV_PATH"])
    log = pd.DataFrame(index=[], columns=[
        'number', 'train', 'val', 'test', 'gaussian_filter_sd', 'sampling', 'NSS', 'CC', 'SIM', 'KLD', 'AUC_Judd', 'dirpath'
    ])
    if not os.path.exists(cfg["DIR"]["ROOT_OUTPUT_DIR"]):
        os.makedirs(cfg["DIR"]["ROOT_OUTPUT_DIR"])

    #TODO tqdmの分母がデータセット全体になっているが、計算しようとしているもののみにする
    for i in tqdm(range(len(filepath_df))):
        train_flag = (cfg["DATA"]["TRAIN"] and filepath_df["train"][i])
        test_flag = (cfg["DATA"]["TEST"] and filepath_df["test"][i])
        val_flag = (cfg["DATA"]["VAL"] and filepath_df["val"][i])

        number = filepath_df["number"][i]
        img_path = filepath_df["img_path"][i]
        hem_path = filepath_df["he_salmap_bin_path"][i]
        scanpath_path = filepath_df["scanpath_path"][i]
        #csvの絶対パスを書き換え
        #img_path = tools.convert_file_path(filepath_df["img_path"][i])
        #hem_path = tools.convert_file_path(filepath_df["he_salmap_bin_path"][i])
        #scanpath_path = tools.convert_file_path(filepath_df["scanpath_path"][i])

        if (train_flag)or(test_flag)or(val_flag):
            salmap_path = os.path.join(cfg["DATA"]["SALMAP_PATH"], "P{}.png".format(number))

            #load posMap
            data = np.ones(cfg["DATA"]["INPUT_WIDTH"]*cfg["DATA"]["INPUT_HEIGHT"])

            if os.path.isfile('{}.npy'.format(hem_path)):
                posMap = np.load('{}.npy'.format(hem_path))
            else:
                # save npy for fast processing next time
                with open(hem_path, "rb") as f:
                    for j in range(cfg["DATA"]["INPUT_WIDTH"]*cfg["DATA"]["INPUT_HEIGHT"]):
                        b = f.read(4)
                        c = struct.unpack("f", b)
                        data[j] = c[0]
                posMap = data.reshape((cfg["DATA"]["INPUT_HEIGHT"], cfg["DATA"]["INPUT_WIDTH"]))
                np.save('{}.npy'.format(hem_path) ,posMap)

            fixations = np.loadtxt(scanpath_path, delimiter=",", skiprows=1, usecols=(1,2))
            fixations = fixations * [cfg["DATA"]["INPUT_WIDTH"], cfg["DATA"]["INPUT_HEIGHT"]] - [1,1]; fixations = fixations.astype(int)

            fixMap = np.zeros(posMap.shape, dtype=int)
            for iFix in range(fixations.shape[0]):
                    fixMap[fixations[iFix, 1], fixations[iFix, 0]] += 1

            #load salMap
            salMap = np.array(Image.open(salmap_path))

            for j, gf in enumerate(gf_list):

                if gf!=0:
                    salMap_gf = filters.gaussian_filter(salMap, gf)
                else:
                    gf = 0
                    salMap_gf = salMap.copy()
                salMap_gf = resize(salMap_gf, (cfg["DATA"]["INPUT_HEIGHT"], cfg["DATA"]["INPUT_WIDTH"]), order=0) #skimage resize nearest neighbor

                for sampling_type in samplig_type_list:
                    # salMap = salMap/salMap.sum()
                    if sampling_type == "Sin":
                        fixMap_sampled = fixMap
                        salMap_sampled = salMap_gf * VerticalWeighting[:, None] + eps
                        posMap_sampled = posMap * VerticalWeighting[:, None] + eps
                    elif sampling_type =="Equi":
                        fixMap_sampled = fixMap
                        salMap_sampled = salMap_gf.copy()
                        posMap_sampled = posMap.copy()
                    elif sampling_type =="Sinusoidal":
                        fixMap_sampled = np.array([equi2sinusoidal_fix(fixations, VerticalWeighting, cfg["DATA"]["INPUT_WIDTH"])])
                        salMap_sampled = np.array([salMap_gf[sinusoidalBinaryMap]])
                        posMap_sampled = np.array([posMap[sinusoidalBinaryMap]])
                    salMap_sampled = normalize(salMap_sampled, method='sum')
                    posMap_sampled = normalize(posMap_sampled, method='sum')

                    # salMap : estimated
                    # posMap : groundtruth

                    kld = KLD(salMap_sampled, posMap_sampled)
                    cc = CC(salMap_sampled, posMap_sampled)
                    nss = NSS(salMap_sampled, fixMap_sampled)
                    aucj = AUC_Judd(salMap_sampled, fixMap_sampled)
                    sim = SIM(salMap_sampled, posMap_sampled)

                    log_tmp = pd.Series([
                        number,
                        train_flag,
                        val_flag,
                        test_flag,
                        gf,
                        sampling_type,
                        nss,
                        cc,
                        sim,
                        kld,
                        aucj,
                        cfg["DATA"]["SALMAP_PATH"]
                    ], index=['number', 'train', 'val', 'test', 'gaussian_filter_sd', 'sampling', 'NSS', 'CC', 'SIM', 'KLD', 'AUC_Judd', 'dirpath'])
                    log = log.append(log_tmp, ignore_index=True)
            log.to_csv(save_path, index=False)
    return save_path

if __name__ == '__main__':
    args = get_parser().parse_args()
    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f)
    main(cfg['EVALUATE'])
