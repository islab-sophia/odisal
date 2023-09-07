import os

import numpy as np
import torch.utils.data as data
import scipy.misc
import configparser
from PIL import Image
import pandas as pd

class TestODIDataset(data.Dataset):
    """
    dataset class for predicting for extracted images from odi

    Parameters
    -----------
    img_files : list, length is (num_odi_files * num_extracted)
        list of path to extracted images
    mean_rgb : list of float
        mean rgb of training dataset from checkpoint
    eqbl_idxs : list, length is same as extracted_files
        list of index of equator bias layer channel used for each extracted image
        ceil's image is 0
        floor's image is (num_vertical-1)
    extraction_idxs : list, length is same as extracted_files
        list of index of extracted image
    size : None or tuple of int, (height, width), (default : None)
        when resizing
        using numpy.resize
        https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.resize.html
    transform : object
        np_transforms module
    """
    def __init__(self, img_files, mean_rgb, eqbl_idxs, extraction_idxs, size=None, transform=None):
        self.img_files = img_files
        self.mean_rgb = mean_rgb
        self.eqbl_idxs = eqbl_idxs
        self.extraction_idxs = extraction_idxs
        self.size = size
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        """
        load and transform image

        Returns
        -----------
        file_name : str
            basename of extracted image
        img : torch.Tensor (if using np_transforms.ToTensor) or ndarray, shape (channels, height, width)
            extracted image
        self.eqbl_indx[index] : int
            index of equator bias layer channel used for extracted image
        self.extraction_idxs[index] : int
            index of extracted image
        """
        img_file = self.img_files[index]

        if self.size==None:
            img = np.array(Image.open(img_file), dtype=np.float32)
        else:
            img = np.array(Image.open(img_file).resize(self.size), dtype=np.float32)
        # img : (h, w, c)
        img -= self.mean_rgb

        if self.transform is not None:
            img = self.transform(img)
        file_name = os.path.basename(img_file).split('.')[0]
        return file_name, img, self.eqbl_idxs[index], self.extraction_idxs[index]

class TestPlanarDataset(data.Dataset):
    """
    dataset class for predicting for planar images

    Parameters
    -----------
    img_files : list, length is (num_odi_files * num_extracted)
        list of path to extracted images
    mean_rgb : list of float
        mean rgb of training dataset from checkpoint
    size : None or tuple of int, (height, width), (default : None)
        when resizing
        using numpy.resize
        https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.resize.html
    transform : object
        np_transforms module
    """
    def __init__(self, img_files, mean_rgb, size=None, transform=None):
        self.img_files = img_files
        self.mean_rgb = mean_rgb
        self.transform = transform
        self.size = size

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        """
        load and transform image

        Returns
        -----------
        file_name : str
            basename of image
        img : torch.Tensor (if using np_transforms.ToTensor) or ndarray, shape (channels, height, width)
            image
        """
        # load and transform image
        img_file = self.img_files[index]

        if self.size==None:
            img = np.array(Image.open(img_file), dtype=np.float32)
        else:
            img = np.array(Image.open(img_file).resize(self.size), dtype=np.float32)
        # img : (h, w, c)
        img -= self.mean_rgb

        if self.transform is not None:
            img = self.transform(img)
        file_name = os.path.basename(img_file).split('.')[0]
        return file_name, img

class SaliencyDataset(data.Dataset):
    #TODO location_target

    def __init__(self, dataset_cfg_path, train=False, val=False, test=False,
                target_type=['location', 'distribution'],
                 pre_simul_transform=None, transform=None,
                 location_target_transform=None,
                 distribution_target_transform=None,
                 post_simul_transform=None):

        config = configparser.ConfigParser()
        config.read(dataset_cfg_path)
        if "dataset_type" in config['dataset'].keys():
            if config['dataset']['dataset_type'] == "txt":
                self.dataset_type = "txt"
            elif config['dataset']['dataset_type'] == "csv":
                self.dataset_type = "csv"
        else:
            self.dataset_type = "csv"
        if "projection_type" in config['dataset'].keys():
            if config['dataset']['projection_type'] == "planar":
                self.projection_type = "planar"
            elif config['dataset']['projection_type'] == "equirectangular":
                self.projection_type = "equirectangular"
        else:
            self.projection_type = "equirectangular"

        if (self.dataset_type == "csv")and(self.projection_type == "equirectangular"):
            self.dataset_name = config['dataset']['dataset_name']
            self.mapping_name = config['dataset']['mapping_name']
            self.parent_dataset_name = config['dataset']['parent_dataset_name']
            self.csv_path = config['dataset']['csv_path']
            mean_r = float(config['dataset']['mean_r'])
            mean_g = float(config['dataset']['mean_g'])
            mean_b = float(config['dataset']['mean_b'])
            self.mean_rgb = np.array([mean_r, mean_g, mean_b])
            self.input_h = int(config['dataset']['input_h'])
            self.input_w = int(config['dataset']['inptu_w'])
            self.extract_h = int(config['dataset']['extract_h'])
            self.extract_w = int(config['dataset']['extract_w'])
            self.num_vertical = int(config['dataset']['num_vertical'])
            self.num_horizontal = int(config['dataset']['num_horizontal'])
            self.view_angle_p = float(config['dataset']['view_angle_p'])
            self.view_angle_t = float(config['dataset']['view_angle_t'])
            filepath_df = pd.read_csv(self.csv_path)
            self.files = []
            for i in range(len(filepath_df)):
                train_flag = (train and filepath_df["train"][i])
                test_flag = (test and filepath_df["test"][i])
                val_flag = (val and filepath_df["val"][i])
                parent_number = filepath_df["parent_number"][i]
                extract_number = filepath_df["extract_number"][i]
                img_path = filepath_df["img_path"][i]
                hem_path = filepath_df["he_salmap_im_path"][i]
                if (train_flag)or(test_flag)or(val_flag):
                    self.files.append({
                        'img': img_path,
                        'location_target': "",
                        'distribution_target': hem_path,
                        'data_id': self.dataset_name+'_'+str(parent_number)+'_'+str(extract_number)
                    })
            self.target_type = target_type
            self.transform = transform
            self.location_target_transform = location_target_transform
            self.distribution_target_transform = distribution_target_transform
            self.pre_simul_transform = pre_simul_transform
            self.post_simul_transform = post_simul_transform

        elif (self.dataset_type == "txt")and(self.projection_type == "planar"):
            self.dataset_name = config["dataset_name"]
            if not isinstance(target_type, (list, tuple)):
                raise TypeError('Type of target_type must be list or tuple.')
            elif len(target_type)==0:
                raise ValueError('len(target_type) must not be 0')
            elif 'location' not in target_type and 'distribution' not in target_type:
                raise ValueError("target_type must be selected from 'location' or 'distribution'")
            imgsets_file = os.path.join(config[imgsets_dir], '{}.txt'.format(data_type))
            files = []
            for data_id in open(imgsets_file).readlines():
                data_id = data_id.strip()
                img_file = os.path.join(config["img_dir"], '{0}{1}'.format(data_id, config["img_tail"]))
                if 'location' not in target_type:
                    location_target_file = ""
                else :
                    location_target_file = os.path.join(config["location_target_dir"], '{0}{1}'.format(data_id, config["location_target_tail"]))
                distribution_target_file = os.path.join(config["distribution_target_dir"], '{0}{1}'.format(data_id, config["distribution_target_tail"]))
                files.append({
                    'img': img_file,
                    'location_target': location_target_file,
                    'distribution_target': distribution_target_file,
                    'data_id': data_id
                })
            self.files = files
            mean_r = float(config['dataset']['mean_r'])
            mean_g = float(config['dataset']['mean_g'])
            mean_b = float(config['dataset']['mean_b'])
            self.mean_rgb = np.array([mean_r, mean_g, mean_b])
            self.num_vertical = config["num_vertical"]
            self.target_type = target_type
            self.transform = transform
            self.location_target_transform = location_target_transform
            self.distribution_target_transform = distribution_target_transform
            self.pre_simul_transform = pre_simul_transform
            self.post_simul_transform = post_simul_transform
            self.num_horizontal = config["num_horizontal"]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        """
        Returns
        -----------
        data : list
            [img, (location_target), (distribution_target), data_id]
        """
        data_file = self.files[index]
        data = []

        img_file = data_file['img']
        img = scipy.misc.imread(img_file, mode='RGB').astype(np.float32)
        img -= self.mean_rgb
        data.append(img)

        # load and transform location target
        if 'location' in self.target_type:
            location_target_file = data_file['location_target']
            location_target = scipy.misc.imread(location_target_file)
            if self.location_target_transform is not None:
                location_target = self.location_target_transform(location_target)
            data.append(location_target)

        # load and transform distribution target
        if 'distribution' in self.target_type:
            distribution_target_file = data_file['distribution_target']
            distribution_target = scipy.misc.imread(distribution_target_file)

            if self.distribution_target_transform is not None:
                distribution_target = self.distribution_target_transform(distribution_target)
            data.append(distribution_target)

        # transform
        if self.pre_simul_transform is not None:
            data = self.pre_simul_transform(*data)
        if self.transform is not None:
            data[0] = self.transform(data[0])
        if 'location' in self.target_type and self.location_target_transform is not None:
            data[1] = self.location_target_transform(data[1])
        if 'distribution' in self.target_type and self.distribution_target_transform is not None:
            data[-1] = self.distribution_target_transform(data[-1])
        if self.post_simul_transform is not None:
            data = self.post_simul_transform(*data)

        data.append(data_file['data_id'])

        return data
