import torch
from torch.utils.data import Dataset

import numpy as np
import torch
from torch.utils.data import Dataset

class PSFDataset(Dataset):
    """A PyTorch Dataset class for handling PSF data.

    This class provides methods for loading and saving datasets, as well as generating datasets from PSFs.

    Attributes:
        data (torch.Tensor): The data tensor.
        labels (torch.Tensor): The labels tensor.
        indices (array-like): The indices array.
        class_names (list of str, optional): The class names. Defaults to indices if not provided.
        split (float, optional): The proportion of data to use for training. Defaults to 0.85.
        device (torch.device): The device where the tensors are stored.
        mode (str): The mode for accessing the dataset, either "train" or "test".
        train_indices (array-like): The indices for training data.
        test_indices (array-like): The indices for testing data.
    """
    def __init__(self, data, labels, indices, class_names=None, split=0.85, device=None):
        """Initializes the PSFDataset.

        Args:
            data (array-like): The PSF data.
            labels (array-like): The labels.
            indices (array-like): The indices for the data.
            class_names (list of str, optional): The class names. Defaults to indices if not provided.
            split (float, optional): The proportion of data to use for training. Defaults to 0.85.
            device (torch.device, optional): The device where the tensors are stored. Defaults to CUDA if available, else CPU.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = torch.from_numpy(data).to(self.device)
        self.labels = torch.from_numpy(labels).long().to(self.device)
        self.indices = indices
        self.class_names = class_names or [str(i) for i in range(labels.max() + 1)]
        self.split = split
        self.mode = "train"

        self.train_indices = indices[:int(split * len(indices))]
        self.test_indices = indices[int(split * len(indices)):]

    def __len__(self):
        """Returns the number of elements in the current mode (train or test).

        Returns:
            int: The number of elements.
        """
        if self.mode == "train":
            return len(self.train_indices)
        else:
            return len(self.test_indices)
    
    def __getitem__(self, idx):
        """Returns a tuple containing the data and label at a given index in the current mode (train or test).

        Args:
            idx (int): The index.

        Returns:
            tuple: A tuple containing the data and label at the given index.
        """
        if self.mode == "train":
            idx = self.train_indices[idx]
        else:
            idx = self.test_indices[idx]
        return self.data[idx].unsqueeze(0).float(), self.labels[idx]
    
    def save(self, path):
        """Saves the dataset to a file.

        Args:
            path (str): The path where the dataset will be saved.
        """
        torch.save({
            'data': self.data.cpu().numpy(),
            'labels': self.labels.cpu().numpy(),
            'indices': self.indices,
            'split': self.split,
            'class_names': self.class_names,
        }, path)
    
    @classmethod
    def load(cls, path, device=None):
        """Loads a dataset from a file.

        Args:
            path (str): The path where the dataset is saved.
            device (torch.device, optional): The device where the tensors are stored. If not provided, the dataset is moved to the correct device when created.

        Returns:
            PSFDataset: The loaded dataset.
        """
        checkpoint = torch.load(path, map_location='cpu')  # load to CPU
        return cls(
            data=checkpoint['data'],
            labels=checkpoint['labels'],
            indices=checkpoint['indices'],
            split=checkpoint['split'],
            class_names=checkpoint['class_names'],
            device=device,  # move to the correct device when creating the dataset
        )
    
    @classmethod
    def create_from_PSFs(cls, categories, class_names=None, device=None, normalize=False, split=0.85):
        """Creates a PSFDataset from given PSFs.

        Args:
            categories (list of array-like): The PSFs for each category.
            class_names (list of str, optional): The class names. If not provided, defaults to indices.
            device (torch.device, optional): The device where the tensors are stored. If not provided, defaults to CUDA if available, else CPU.
            normalize (bool, optional): Whether to normalize the data. Defaults to False.
            split (float, optional): The proportion of data to use for training. Defaults to 0.85.

        Returns:
            PSFDataset: The created dataset.
        """
        # Get the minimum class size
        min_class_size = min(len(category) for category in categories)
        random_seed = 1 # random seed for reproducibility
        np.random.seed(random_seed)
        # Randomly sample instances from each class to equalize class sizes
        equalized_categories = [category[np.random.randint(category.shape[0], size=min_class_size), ...] 
                            for category in categories]
        # Assign data to classes
        if normalize:
            normalized = np.array([cls.normalize(image) for image in np.concatenate(equalized_categories)])
            data = normalized.astype(np.float16)
        else:
            data = np.concatenate(equalized_categories).astype(np.int32)
        labels = np.concatenate([np.full(min_class_size, i) for i, _ in enumerate(categories)])
        indices = np.random.choice(np.arange(len(data)), size=len(data), replace=False)
        return cls(data, labels, indices, class_names=class_names, split=split, device=device)
    
    @classmethod
    def normalize(cls, image):
        """Normalizes a given image to the 0-1 range.

        Args:
            image (array-like): The image to be normalized.

        Returns:
            array-like: The normalized image.
        """
        image_min = np.min(image)
        image_max = np.max(image)
        image_range = image_max - image_min
        normalized = (image - image_min) / image_range
        return normalized
    
    def train(self):
        """Sets the mode to "train"."""
        self.mode = "train"
    
    def test(self):
        """Sets the mode to "test"."""
        self.mode = "test"
