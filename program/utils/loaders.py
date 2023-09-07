import torch

class MultiScaleTestODILoader(object):
    """
    loader for predicting for extracted images from odi

    Parameters
    -----------
    dataset : object
        dataset class's instance object
    batch_size : int (default : 1)
        batch size
    """

    def __init__(self, dataset, batch_size=1):
        self._dataset = dataset
        self.batch_size = batch_size
        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        """
        Returns
        -----------
        file_names : list, length is batch_size
            list of basename of extracted images
        tensors : torch.Tensor, shape (batch_size, channels, height, width)
            tensors of extracted images
        eqbl_idxs : list, length is batch_size
            list of index of equator bias layer channel used for each extracted image
            ceil's image is 0
            floor's image is (num_eqbl_channels-1)
        extraction_idxs : list, length is batch_size
            list of index of extracted image

        these are created with basemapping.load_extract_save
        """
        file_name, tensors_list, eqbl_idx, extraction_idx = self._dataset[0]
        file_names = []
        tensors = [[] for i in range(len(tensors_list))]
        eqbl_idxs = []
        extraction_idxs = []
        count = 0
        for num in range(self.batch_size):
            if self._i + num == len(self._dataset):
                raise StopIteration()
            file_name, tensors_list, eqbl_idx, extraction_idx = self._dataset[self._i+num]
            for i in range(len(tensors_list)):
                tensors_list[i].unsqueeze_(0)
                tensors[i].append(tensors_list[i])
                
            file_names.append(file_name)
            eqbl_idxs.append(eqbl_idx)
            extraction_idxs.append(extraction_idx)
            count += 1

            if self._i + num == len(self._dataset) - 1:
                break

        self._i += count
        
        for i in range(len(tensors)):
            tensors[i] = torch.cat(tensors[i], dim=0)
        return (file_names, tensors_list, eqbl_idxs, extraction_idxs)

    next = __next__

class TestODILoader(object):
    """
    loader for predicting for extracted images from odi

    Parameters
    -----------
    dataset : object
        dataset class's instance object
    batch_size : int (default : 1)
        batch size
    """

    def __init__(self, dataset, batch_size=1):
        self._dataset = dataset
        self.batch_size = batch_size
        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        """
        Returns
        -----------
        file_names : list, length is batch_size
            list of basename of extracted images
        tensors : torch.Tensor, shape (batch_size, channels, height, width)
            tensors of extracted images
        eqbl_idxs : list, length is batch_size
            list of index of equator bias layer channel used for each extracted image
            ceil's image is 0
            floor's image is (num_eqbl_channels-1)
        extraction_idxs : list, length is batch_size
            list of index of extracted image

        these are created with basemapping.load_extract_save
        """
        file_names = []
        tensors = []
        eqbl_idxs = []
        extraction_idxs = []
        count = 0
        for num in range(self.batch_size):
            if self._i + num == len(self._dataset):
                raise StopIteration()
            file_name, tensor, eqbl_idx, extraction_idx = self._dataset[self._i+num]
            tensor.unsqueeze_(0)
            file_names.append(file_name)
            tensors.append(tensor)
            eqbl_idxs.append(eqbl_idx)
            extraction_idxs.append(extraction_idx)
            count += 1

            if self._i + num == len(self._dataset) - 1:
                break

        self._i += count

        tensors = torch.cat(tensors, dim=0)
        return (file_names, tensors, eqbl_idxs, extraction_idxs)

    next = __next__
    
class TestPlanarLoader(object):
    """
    loader for predicting for planar images

    Parameters
    -----------
    dataset : object
        dataset class's instance object
    batch_size : int (default : 1)
        batch size
    """
    def __init__(self, dataset, batch_size=1):
        self._dataset = dataset
        self.batch_size = batch_size
        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        """
        Returns
        -----------
        file_names : list, length is batch_size
            list of basename of images
        tensors : torch.Tensor, shape (batch_size, channels, height, width)
            tensors of images
        """

        file_names = []
        tensors = []
        count = 0
        for num in range(self.batch_size):
            if self._i + num == len(self._dataset):
                raise StopIteration()
            file_name, tensor = self._dataset[self._i+num]
            tensor.unsqueeze_(0)
            file_names.append(file_name)
            tensors.append(tensor)
            count += 1

            if self._i + num == len(self._dataset) - 1:
                break

        self._i += count

        tensors = torch.cat(tensors, dim=0)

        return (file_names, tensors)

    next = __next__


def TrainLoader(dataset, batch_size=1, workers=1):
    """
    train and validation loader

    Parameters
    -----------
    dataset : object
        dataset class's instance object
    batch_size : int (default : 1)
        batch size
    workers : int (default : 1)
        num of workers

    Returns
    -----------
    train_loader : object
    """
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, sampler=None)
    return train_loader
