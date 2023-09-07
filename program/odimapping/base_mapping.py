from PIL import Image
import numpy as np
import os
import imageio
import gc

import torch
import torch.nn.functional as F

from skimage.transform import resize

class BaseMapping():
    """
    class for mapping from ODI to 2D image and vice versa

    Parameters
    -----------
    camera_list : list
    input_method : str
    name : str
    embed_overlapping : str

    extract_size : tuple of int
        (extract_h, extract_w)
    odi_size : tuple of int
        (odi_h, odi_w)
    odi_c : int
    view_angle : tuple of float
        radians, (view_angle_p, view_angle_t)
    num_vertical : int
    num_extracted : int
    num_horizontal : int
    maps_dir : str
    """
    def __init__(
            self, camera_list,input_method, name,
            embed_overlapping, extract_size,
            odi_size, odi_c, view_angle,
            num_vertical, num_extracted,
            num_horizontal, maps_dir):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.camera_list = np.radians(camera_list)
        self.input_method = input_method
        self.name = name
        self.embed_overlapping = embed_overlapping
        self.extract_size = extract_size
        self.odi_size = odi_size
        self.odi_c = odi_c
        self.view_angle = view_angle
        self.num_vertical = num_vertical
        self.num_extracted = num_extracted
        self.num_horizontal = num_horizontal

        self.maps_path = os.path.join(maps_dir, input_method+name+'.npz')
        if os.path.isfile(self.maps_path):
            self.maps = np.load(self.maps_path)
            # Calculate again when settings are changed.
            load_odi_size = (self.maps['odi_size'][0], self.maps['odi_size'][1])
            load_extract_size = (self.maps['extract_size'][0], self.maps['extract_size'][1])
            load_view_angle = (self.maps['view_angle'][0], self.maps['view_angle'][1])
            if (load_odi_size!=self.odi_size)or(self.maps['odi_c']!=self.odi_c)or(load_extract_size!=self.extract_size)or(load_view_angle!=self.view_angle):
                self.maps = self._create_maps()
        else:
            if not os.path.exists(maps_dir):
                os.makedirs(maps_dir)
            self.maps = self._create_maps()

    def _create_maps(self):
        """
        create maps file for extracting from odi and embedding in odi

        Returns
        -----------
        maps : dict
            dictionary
            'extract' : ndarray (np.int16), shape (num_extracted, 2, height, width)
                extract_map
                axis=1 is [r (y), c (x)]
            'embed' : ndarray (np.int16), shape (num_extracted, 2, height, width)
                embed_map
                axis=1 is [r (y), c (x)]
            'mask' : ndarray (np.int16), shape (num_extracted, height, width)
                mask
            'num' : int
                num_extracted
        """
        print('create maps ...')

        L = (np.array(self.extract_size) / 2.0) / np.tan(np.array(self.view_angle) / 2.0)

        if rnd(L[0]) != rnd(L[1]):
            print('Warning: extract_size and view_angle are inappropriate.') # todo
            return
        else:
            self.L = L[0]
            
        num_extracted = len(self.camera_list)

        for i, camera_angle in enumerate(self.camera_list):

            # Camera direction
            self.nc = np.array([
                    np.cos(camera_angle[0]) * np.cos(camera_angle[1]),
                    -np.cos(camera_angle[0]) * np.sin(camera_angle[1]),
                    np.sin(camera_angle[0])
                ])

            # Distance to the center of the image
            self.c0 = self.L * self.nc

            self.xn = np.array([
                    -np.sin(camera_angle[1]),
                    -np.cos(camera_angle[1]),
                    0
                ])
            self.yn = np.array([
                    -np.sin(camera_angle[0]) * np.cos(camera_angle[1]),
                    np.sin(camera_angle[0]) * np.sin(camera_angle[1]),
                    np.cos(camera_angle[0])
                ])

            #------------
            # extract

            [c1, r1] = np.meshgrid(np.arange(0, rnd(self.extract_size[1])), np.arange(0, rnd(self.extract_size[0])))
            # c1 : x's ndarray (height, width), r1 : y's ndarray (height, width)

            img_cord = [-r1 + self.extract_size[0] / 2.0, c1 - self.extract_size[1] / 2.0]

            self.p = self._get_3Dcordinate(img_cord[0], img_cord[1])
            self.polar_omni_cord = polar(self.p)

            # 2d-cordinates in omni-directional image [c2, r2]
            c2 = (self.polar_omni_cord[1] / (2.0 * np.pi) + 1.0 / 2.0) * (self.odi_size[1] - 1)
            r2 = (-self.polar_omni_cord[0] / np.pi + 1.0/2.0) * (self.odi_size[0] - 1)

            if i==0:
                extract_map = np.zeros((len(self.camera_list), 2, c2.shape[0], c2.shape[1]), dtype=np.int16)
            extract_map[i, 0] = rnd(r2)
            extract_map[i, 1] = rnd(c2)

            #---------------
            # embed

            [c_omni, r_omni] = np.meshgrid(np.arange(self.odi_size[1]), np.arange(self.odi_size[0]))
            theta = (2.0 * c_omni / float(self.odi_size[1]-1) - 1.0) * np.pi #theta.shape=(800, 1600)
            phi = (0.5 - r_omni / float(self.odi_size[0]-1)) * np.pi #phi.shape=(800, 1600)
            pn = np.array([
                    np.cos(phi) * np.cos(theta),
                    -np.cos(phi) * np.sin(theta),
                    np.sin(phi)
                ]) #pn.shape=(3, 800, 1600)
            pn = pn.transpose(1,2,0) #pn.shape=(800, 1600, 3)

            # True: inside image (candidates), False: outside image
            cos_alpha = np.dot(pn, self.nc)
            self.mask = cos_alpha >= 2 * self.L / np.sqrt(self.extract_size[1]**2 + self.extract_size[0]**2 + 4*self.L**2) # circle

            r = np.zeros((self.odi_size[0], self.odi_size[1]))
            xp = np.zeros((self.odi_size[0], self.odi_size[1]))
            yp = np.zeros((self.odi_size[0], self.odi_size[1]))

            r[self.mask == True] = self.L / np.dot(pn[self.mask == True], self.nc)
            xp[self.mask == True] = r[self.mask == True] * np.dot(pn[self.mask == True], self.xn)
            yp[self.mask == True] = r[self.mask == True] * np.dot(pn[self.mask == True], self.yn)

            self.mask = (self.mask == True) & (xp > -self.extract_size[1]/2.0) & (xp < self.extract_size[1]/2.0) & (yp > -self.extract_size[0]/2.0) & (yp < self.extract_size[0]/2.0)
            r[self.mask == False] = 0
            xp[self.mask == False] = 0
            yp[self.mask == False] = 0

            # 2D cordinates in extracted image
            [r1, c1] = np.array([(self.extract_size[0] - 1) / 2.0 - yp, (self.extract_size[1] - 1) / 2.0 + xp]) * self.mask

            #r1_intとc1_intの値を一定区間内に制限
            [r1_int, c1_int] = [rnd(r1), rnd(c1)]
            r1_int = self._limit_values(r1_int, [0, self.extract_size[0]-1], 0)
            c1_int = self._limit_values(c1_int, [0, self.extract_size[1]-1], 0)

            if i==0:
                embed_map = np.zeros((len(self.camera_list), 2, r1_int.shape[0], r1_int.shape[1]), dtype = np.int16)
                mask = np.zeros((len(self.camera_list), self.mask.shape[0], self.mask.shape[1]), dtype = np.int16) #np.float32?
            embed_map[i, 0] = r1_int
            embed_map[i, 1] = c1_int
            mask[i] = self.mask
            #-------------------

        np.savez(self.maps_path, extract = extract_map, embed = embed_map, mask = mask, num = num_extracted,
                odi_size = self.odi_size, odi_c = self.odi_c,
                extract_size = self.extract_size, view_angle = self.view_angle)
        maps = {'extract':extract_map, 'embed':embed_map,
                'mask':mask, 'num':num_extracted,
                'odi_size':self.odi_size, 'odi_c':self.odi_c,
                'extract_size':self.extract_size,
                'view_angle':self.view_angle
                }

        del extract_map, embed_map, mask, num_extracted
        del r1_int, c1_int, r1, c1, r, xp, yp, c_omni, r_omni, theta, phi, pn, c2, r2, img_cord, cos_alpha
        gc.collect()

        print('Done')

        return maps

    def _limit_values(self, x, r, xcopy=1):
        """
        limit values

        Parameters
        -----------
        x : ndarray
        r : list of [lower, upper]

        Returns
        -----------
        ret : ndarray
        """
        if xcopy == 1:
            ret = x.copy()
        else:
            ret = x
        ret[ret<r[0]] = r[0]
        ret[ret>r[1]] = r[1]
        return ret

    def _get_3Dcordinate(self, yp, xp):
        if type(xp) is np.ndarray: # xp, yp: array
            return xp * self.xn.reshape((3,1,1)) + yp * self.yn.reshape((3,1,1)) + np.ones(xp.shape) * self.c0.reshape((3,1,1))
        else: # xp, yp: scalars
            return xp * self.xn + yp * self.yn + self.c0

    def save_before_embedding(self, outputs, save_dir, file_names):
        save_files = []
        length = outputs.size(0)
        odi_max = 0
        for num in range(length):
            saliencymap = outputs.data[num].cpu().numpy()
            odi_max = max(odi_max, saliencymap.max())
        for num in range(length):
            saliencymap = outputs.data[num].cpu().numpy()
            saliencymap = np.uint8(255*saliencymap/odi_max)
            save_path = os.path.join(save_dir, file_names[num]+'.png')
            imageio.imwrite(save_path, saliencymap.transpose(1,2,0))
            save_files.append(save_path)
        return save_files

    def load_embed(
            self, save_files,use_constant_equator_bias=False,
            constant_equator_bias_path=None):
        """
        embed extracted images in odi

        Parameters
        -----------
        save_files : list
            path to extracted images
        use_constant_equator_bias : bool
        constant_equator_bias_path : str

        Returns
        -----------
        odi_ndarray : ndarray, shape(height, width)
        """

        extracted_ndarray = [np.array(Image.open(save_files[num]).resize(self.extract_size)) for num in range(len(save_files))]
        extracted_ndarray = np.stack(extracted_ndarray)
        extracted_ndarray = extracted_ndarray.astype(np.float32)
        # extracted_ndarray : (num_extracted, channels, height, width)
        odi_ndarray = self.embed(extracted_ndarray)

        if use_constant_equator_bias:
            # method A
            constant_equator_bias = np.load(constant_equator_bias_path)
            constant_equator_bias = resize(constant_equator_bias, (odi_ndarray.shape[0], odi_ndarray.shape[1]), order=0, mode="constant")
            odi_ndarray = odi_ndarray*constant_equator_bias[:, :]

        odi_ndarray = np.uint8(255*odi_ndarray/odi_ndarray.max())
        return odi_ndarray

    def load_extract_save(self, odi_files):
        """
        loading odi, extracting 2D images from odi and saving 2D images.

        Parameters
        -----------
        odi_files : list
            list of path to ODI files

        Returns
        -----------
        extracted_files : list, length is (num_odi_files * num_extracted)
            list of path to extracted images
        eqbl_idxs : list, length is same as extracted_files
            list of index of equator bias layer channel used for each extracted image
            ceil's image is 0
            floor's image is (num_vertical-1)
        extraction_idxs : list, length is same as extracted_files
            list of index of extracted image
        """
        # make save-dir
        save_dir = os.path.join(os.path.dirname(odi_files[0]),'ODIextraction',self.name)
        if not(os.path.exists(save_dir)):
            os.makedirs(save_dir)

        # loop per odi
        extracted_files = []
        eqbl_idxs = []
        extraction_idxs = []
        for odi_file in odi_files:
            odi_ndarray = np.expand_dims(np.array(Image.open(odi_file).resize((self.odi_size[1],self.odi_size[0])), dtype=np.float32), axis=0)
            extracted_ndarray = self.extract(odi_ndarray)

            # loop per extracted image
            for extraction_idx in range(extracted_ndarray.shape[0]):
                extracted_im = Image.fromarray(np.uint8(extracted_ndarray[extraction_idx]))
                file_name = os.path.splitext(os.path.basename(odi_file))[0]+'_'+str(extraction_idx)+'.jpg'
                file_name = os.path.join(save_dir, file_name)
                extracted_im.save(file_name)
                extracted_files.append(file_name)

                if extraction_idx == 0 :
                    # ceil's eqbl_idx
                    eqbl_idx = 0
                elif extraction_idx ==1 :
                    # floor's eqbl_idx
                    eqbl_idx = self.num_vertical-1
                else:
                    # others' eqbl_idx
                    eqbl_idx = np.ceil((extraction_idx-1)/self.num_horizontal).astype(np.int)
                eqbl_idxs.append(eqbl_idx)
                extraction_idxs.append(extraction_idx)

        return extracted_files, eqbl_idxs, extraction_idxs
    
    def load_extract_save_(self, odi_files, view_angle):
        """
        loading odi, extracting 2D images from odi and saving 2D images.

        Parameters
        -----------
        odi_files : list
            list of path to ODI files

        Returns
        -----------
        extracted_files : list, length is (num_odi_files * num_extracted)
            list of path to extracted images
        eqbl_idxs : list, length is same as extracted_files
            list of index of equator bias layer channel used for each extracted image
            ceil's image is 0
            floor's image is (num_vertical-1)
        extraction_idxs : list, length is same as extracted_files
            list of index of extracted image
        """
        # make save-dir
        save_dir = os.path.join(os.path.dirname(odi_files[0]),'ODIextraction',self.name)
        if not(os.path.exists(save_dir)):
            os.makedirs(save_dir)

        # loop per odi
        extracted_files = []
        eqbl_idxs = []
        extraction_idxs = []
        for odi_file in odi_files:
            odi_ndarray = np.expand_dims(np.array(Image.open(odi_file).resize((self.odi_size[1],self.odi_size[0])), dtype=np.float32), axis=0)
            extracted_ndarray = self.extract(odi_ndarray)

            # loop per extracted image
            for extraction_idx in range(extracted_ndarray.shape[0]):
                extracted_im = Image.fromarray(np.uint8(extracted_ndarray[extraction_idx]))
                file_name = os.path.splitext(os.path.basename(odi_file))[0]+'_'+str(extraction_idx)+'.jpg'
                file_save_dir = os.path.join(save_dir, str(view_angle))
                if not(os.path.exists(file_save_dir)):
                    os.makedirs(file_save_dir)
                file_name = os.path.join(save_dir, str(view_angle), file_name)
                extracted_im.save(file_name)
                extracted_files.append(file_name)

                if extraction_idx == 0 :
                    # ceil's eqbl_idx
                    eqbl_idx = 0
                elif extraction_idx ==1 :
                    # floor's eqbl_idx
                    eqbl_idx = self.num_vertical-1
                else:
                    # others' eqbl_idx
                    eqbl_idx = np.ceil((extraction_idx-1)/self.num_horizontal).astype(np.int)
                eqbl_idxs.append(eqbl_idx)
                extraction_idxs.append(extraction_idx)

        return extracted_files, eqbl_idxs, extraction_idxs

    def extract(self, odi_ndarray):
        """
        extract 2D image from ODI

        Parameters
        -----------
        odi_ndarray : ndarray (1, height, width, channels)
            ODI's ndarray

        Returns
        -----------
        extracted_ndarray : ndarray (num_extracted_images, heihgt, width, channels)
            Extracted images' ndarray
        """

        # odi_ndarray : 1, h, w, c

        # extract using maps
        extracted_ndarray = [odi_ndarray[0][self.maps['extract'][num, 0], self.maps['extract'][num, 1]] for num in range(self.maps['num'])]
        # extracted_ndarray : list, ndarray (h, w , c) * num

        extracted_ndarray = np.stack(extracted_ndarray)
        # extracted_ndarray: num, h, w, c

        return extracted_ndarray

    def embed(self, extracted_ndarray):
        """
        embed extracted images in ODI

        Parameters
        -----------
        extracted_ndarray : ndarray, shape (num_extracted, channels, height, width)

        Returns
        -----------
        odi_ndarray : ndarray, shape (height, width)
        """
        odi_ndarray = [extracted_ndarray[num][self.maps['embed'][num, 0], self.maps['embed'][num, 1]] for num in range(self.maps['num'])]
        # odi_ndarray : list, ndarray(height, width, channels)*extracted_num

        # masking
        for num in range(self.maps['num']):
            if len(odi_ndarray[num].shape) == 3:
                mask = self.maps['mask'][num].reshape(self.maps['mask'][num].shape[0], self.maps['mask'][num].shape[1], -1)
            else:
                mask = self.maps['mask'][num]
            mask = mask.astype(np.float32)
            if num==0:
                mask_sum = np.copy(mask)
            else:
                mask_sum += mask
            # mask_torch: extracted_num, h, w
            odi_ndarray[num] = odi_ndarray[num]*mask
        odi_ndarray = np.stack(odi_ndarray)
        # odi_ndarray: extracted_num, h, w

        # merge into one image
        if self.embed_overlapping=="max":
            odi_ndarray = np.amax(odi_ndarray, axis=0)
        elif self.embed_overlapping=="average":
            """
            recommended method used in method of paper
            """
            input_sum = np.sum(odi_ndarray, axis=0)
            odi_ndarray = input_sum/mask_sum
        # odi_ndarray : h, w

        return odi_ndarray

def rnd(x):
    """
    rounding float or ndarray
    Parameters
    -----------
    x : float or ndarray
    Returns
    -----------
    int or ndarray (np.int)
    """
    if type(x) is np.ndarray:
        return (x+0.5).astype(np.int)
    else:
        return round(x)


def polar(cord):
    """
    polar cordinate
    Parameters
    -----------
    cord : ndarray,
        cord.shape (3, c1, r1), cord[:,0,0]=[px, py, pz]
        cord.shape = (3,), cord=[px, py, pz]
    Returns
    -----------
    list of [phi, theta]
    phi : ndarray
    theta : ndarray
    """
    if cord.ndim == 1:
        P = np.linalg.norm(cord) # calculate norm
    else:
        P = np.linalg.norm(cord, axis=0) #axis=0: norm of column vectors; axis=1: norm of row vectors
    phi = np.arcsin(cord[2] / P)
    theta_positive = np.arccos(cord[0] / np.sqrt(cord[0]**2 + cord[1]**2))
    theta_negative = - np.arccos(cord[0] / np.sqrt(cord[0]**2 + cord[1]**2))
    theta = (cord[1] > 0) * theta_negative + (cord[1] <= 0) * theta_positive
    return [phi, theta]
