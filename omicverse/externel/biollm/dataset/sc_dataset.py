#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: sc_dataset.py
@time: 2025/3/25 15:13
"""
from torch.utils.data import Dataset
import torch
from scipy.sparse import issparse
from typing import Dict, Tuple
from torch import Tensor


class ScbertDataset(Dataset):
    """
    Custom PyTorch Dataset for single-cell transcriptomics data.

    This dataset class is designed to process single-cell data, where the input is either sparse or dense.
    It performs binning by setting values higher than `bin_num - 2` to `bin_num - 2`. It also handles label
    information for supervised learning tasks.

    Args:
        data (scipy.sparse.spmatrix or numpy.ndarray): The input data, either sparse or dense, where each row
            represents a data sample (e.g., a cell).
        bin_num (int): The number of bins for binning the values in the data. Values higher than `bin_num - 2` are
            capped at `bin_num - 2`.
        label (numpy.ndarray, optional): The labels corresponding to each sample in `data`. Default is `None`.

    Attributes:
        data (scipy.sparse.spmatrix or numpy.ndarray): The input data.
        label (numpy.ndarray or None): The labels for each sample, or `None` if labels are not provided.
        bin_num (int): The number of bins for data binning.

    Methods:
        __getitem__(index):
            Returns a data sample (with binning applied) and its corresponding label, if available.

        __len__():
            Returns the number of samples in the dataset.
    """

    def __init__(self, data, bin_num, label=None):
        """
        Initializes the ScbertDataset with the given data, binning number, and optional labels.

        Args:
            data (scipy.sparse.spmatrix or numpy.ndarray): Input data.
            bin_num (int): Number of bins for data binning.
            label (numpy.ndarray, optional): Labels for the data. Defaults to None.
        """
        super().__init__()
        self.data = data
        self.label = label
        self.bin_num = bin_num

    def __getitem__(self, index):
        """
        Retrieves a single data sample and its label, with data binning applied.

        Args:
            index (int): The index of the data sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - `full_seq` (torch.Tensor): The data sample, binned and converted to a PyTorch tensor.
                - `seq_label` (numpy.ndarray or None): The label for the data sample, or `None` if no label is provided.
        """
        rand_start = index
        # rand_start = random.randint(0, self.data.shape[0]-1)
        full_seq = self.data[rand_start].toarray()[0] if issparse(self.data) else self.data[rand_start]
        full_seq[full_seq > (self.bin_num - 2)] = self.bin_num - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0])))
        if self.label is not None:
            seq_label = self.label[rand_start]
            return full_seq, seq_label
        else:
            return full_seq

    def __len__(self):
        """
        Returns the number of data samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return self.data.shape[0]


class ScgptDataset(Dataset):
    """
    A custom dataset class for handling single-cell transcriptomics data for use in the ScGPT model.

    This class is designed to handle data in a dictionary format, where each key represents a different data
    element (e.g., gene IDs, expression values, etc.). It allows for easy indexing and batching of data for
    model training or inference.

    Args:
        data (Dict[str, torch.Tensor]): A dictionary containing data for the dataset. The dictionary should have
                                         keys as strings and values as tensors (e.g., "gene_ids" -> torch.Tensor).

    Attributes:
        data (Dict[str, torch.Tensor]): The data dictionary containing different tensors that represent the dataset.

    Methods:
        __len__():
            Returns the number of data points (i.e., the length of the dataset).

        __getitem__(idx):
            Retrieves the data at the given index `idx` and returns it as a dictionary of tensors.
    """

    def __init__(self, data: Dict[str, torch.Tensor]):
        """
        Initializes the ScgptDataset with a dictionary of tensors.

        Args:
            data (Dict[str, torch.Tensor]): A dictionary where keys are data elements (e.g., "gene_ids") and
                                             values are tensors corresponding to those data elements.
        """
        self.data = data

    def __len__(self):
        """
        Returns the number of data points in the dataset.

        Returns:
            int: The length of the dataset, which corresponds to the number of data points (i.e., the length of "gene_ids").
        """
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        """
        Retrieves the data at the specified index `idx`.

        Args:
            idx (int): The index of the data point to retrieve.

        Returns:
            dict: A dictionary where the keys are data elements (e.g., "gene_ids") and the values are the tensors
                  corresponding to those elements for the given index `idx`.
        """
        return {k: v[idx] for k, v in self.data.items()}

class GeneformerDataset(Dataset):
    """
    A custom dataset class for handling single-cell transcriptomics data for use in the Geneformer model.

    This class is designed to handle data in a dictionary format, where each key represents a different data
    element (e.g., gene IDs, expression values, etc.). It allows for easy indexing and batching of data for
    model training or inference.

    Args:
        data (Dict[str, torch.Tensor]): A dictionary containing data for the dataset. The dictionary should have
                                         keys as strings and values as tensors (e.g., "input_ids" -> torch.Tensor).

    Attributes:
        data (Dict[str, torch.Tensor]): The data dictionary containing different tensors that represent the dataset.

    Methods:
        __len__():
            Returns the number of data points (i.e., the length of the dataset).

        __getitem__(idx):
            Retrieves the data at the given index `idx` and returns it as a dictionary of tensors.
    """

    def __init__(self, data: Dict[str, torch.Tensor]):
        """
        Initializes the ScgptDataset with a dictionary of tensors.

        Args:
            data (Dict[str, torch.Tensor]): A dictionary where keys are data elements (e.g., "input_ids") and
                                             values are tensors corresponding to those data elements.
        """
        self.data = data

    def __len__(self):
        """
        Returns the number of data points in the dataset.

        Returns:
            int: The length of the dataset, which corresponds to the number of data points (i.e., the length of "input_ids").
        """
        return self.data["length"]

    def __getitem__(self, idx):
        """
        Retrieves the data at the specified index `idx`.

        Args:
            idx (int): The index of the data point to retrieve.

        Returns:
            dict: A dictionary where the keys are data elements (e.g., "gene_ids") and the values are the tensors
                  corresponding to those elements for the given index `idx`.
        """
        return {k: v[idx] for k, v in self.data.items()}


class ScfoundationDataset(Dataset[Tuple[Tensor, ...]]):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """
    tensors: Tuple[Tensor, ...]

    def __init__(self, *tensors: Tensor) -> None:
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)