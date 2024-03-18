import torch
from torch.utils.data import Dataset
import numpy as np

class StandardDataset(Dataset):
    """A baseline PyTorch Dataset class.

    This class provides methods for loading and saving datasets.

    Attributes:
        data (torch.Tensor): The data tensor.
        labels (torch.Tensor): The labels tensor.
        indices (list): List of all indices.
        train_indices (list): List of training indices.
        test_indices (list): List of testing indices.
        class_names (list of str, optional): The class names. Defaults to indices if not provided.
        device (torch.device): The device where the tensors are stored.
    """
    def __init__(self, data, labels, split, class_names=None, device=None):
        """Initializes the REPDataset.

        Args:
            data (array-like): The input data.
            labels (array-like): The labels.
            split (float): The ratio of training data to total data.
            class_names (list of str, optional): The class names. Defaults to indices if not provided.
            device (torch.device, optional): The device where the tensors are stored. Defaults to CUDA if available, else CPU.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = torch.from_numpy(data).to(self.device)
        self.labels = torch.from_numpy(labels).long().to(self.device)
        self.class_names = class_names or [str(i) for i in range(labels.max() + 1)]
        
        self.indices = np.arange(len(data))
        random_seed = 1
        np.random.seed(random_seed)
        np.random.shuffle(self.indices)

        train_size = int(len(self.indices) * split)
        self.train_indices = self.indices[:train_size]
        self.test_indices = self.indices[train_size:]

    def __len__(self):
        """Returns the total number of elements in the dataset.

        Returns:
            int: The total number of elements.
        """
        return len(self.indices)

    def __getitem__(self, idx):
        """Returns a tuple containing the data and label at a given index.

        Args:
            idx (int): The index.

        Returns:
            tuple: A tuple containing the data and label at the given index.
        """
        data_idx = self.indices[idx]
        return self.data[data_idx].unsqueeze(0).float(), self.labels[data_idx]
    
    def save(self, path):
        """Saves the dataset to a file.

        Args:
            path (str): The path where the dataset will be saved.
        """
        torch.save({
            'data': self.data.cpu().numpy(),
            'labels': self.labels.cpu().numpy(),
            'indices': self.indices,
            'train_indices': self.train_indices,
            'test_indices': self.test_indices,
            'split': self.split,
            'class_names': self.class_names,
        }, path)
    
    @classmethod
    def load(cls, path, device=None):
        checkpoint = torch.load(path, map_location='cpu')  # load to CPU
        dataset = cls(
            data=checkpoint['data'],
            labels=checkpoint['labels'],
            split=checkpoint['split'],
            class_names=checkpoint['class_names'],
            device=device,  # move to the correct device when creating the dataset
        )
        dataset.indices = checkpoint['indices']
        dataset.train_indices = checkpoint['train_indices']
        dataset.test_indices = checkpoint['test_indices']
        return dataset