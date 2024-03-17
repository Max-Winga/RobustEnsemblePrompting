import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

class REPDataset(Dataset):
    """A PyTorch Dataset class which adds layers of perturbation to data.

    This class provides methods for loading and saving datasets.

    Attributes:
        data (torch.Tensor): The data tensor.
        labels (torch.Tensor): The labels tensor.
        indices (array-like): The indices array.
        class_names (list of str, optional): The class names. Defaults to indices if not provided.
        device (torch.device): The device where the tensors are stored.
        mode (str): The mode for accessing the dataset, either "train" or "test".
        train_indices (array-like): The indices for training data.
        test_indices (array-like): The indices for testing data.
        perturbations (list): A list of perturbation functions which add noise to a single piece of input data.
        multiplicity (int): The multiplicity of the output dimension compared to the input.
        include_original (bool): Whether to include the raw datapoint in the perturbations.
        shuffle (bool): Whether to shuffle the order of the perturbations in the stack.
    """
    def __init__(self, data, labels, indices, train_indices, test_indices, class_names=None, 
                 device=None, perturbations=[], include_original=True, shuffle=False):
        """Initializes the REPDataset.

        Args:
            data (array-like): The input data.
            labels (array-like): The labels.
            indices (array-like): The indices for the data.
            train_indices (array-like): The indices for training data.
            test_indices (array-like): The indices for testing data.
            class_names (list of str, optional): The class names. Defaults to indices if not provided.
            device (torch.device, optional): The device where the tensors are stored. Defaults to CUDA if available, else CPU.
            perturbations (list): A list of perturbation functions which add noise to a single piece of input data.
            include_original (bool): Whether to include the raw datapoint in the perturbations.
            shuffle (bool): Whether to shuffle the order of the perturbations in the stack.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = torch.from_numpy(data).to(self.device)
        self.labels = torch.from_numpy(labels).long().to(self.device)
        self.indices = indices
        self.class_names = class_names or [str(i) for i in range(labels.max() + 1)]
        self.mode = "train"
        
        self.train_indices = train_indices
        self.test_indices = test_indices
        
        self.perturbations = perturbations
        self.include_original = include_original
        self.shuffle = shuffle
        self.multiplicity = len(self.perturbations) + int(self.include_original)


    def __len__(self):
        """Returns the number of elements in the current mode (train or test).

        Returns:
            int: The number of elements.
        """
        if self.mode == "train":
            return len(self.train_indices)
        else:
            return len(self.test_indices)
        
    def __getitem__(self, idx, raw=False):
        """Returns a tuple containing the data and label at a given index.

        Args:
            idx (int): The index.
            raw (bool, optional): If True, return the raw datapoint, else the multiply-perturbed version. Defaults to False.

        Returns:
            tuple: A tuple containing the data and label at the given index.
        """
        raw_data = self.data[idx].unsqueeze(0).float()
        if raw:
            return raw_data, self.labels[idx]
        else:
            layers = [raw_data]
            for perturbation in self.perturbations:
                layers.append(perturbation(raw_data))

            if self.include_original:
                perturbed_data = torch.stack(layers, dim=0)
            else:
                perturbed_data = torch.stack(layers[1:], dim=0)

            if self.shuffle:
                # Generate a random permutation of indices
                permutation = torch.randperm(perturbed_data.size(0))
                # Shuffle the stacked tensor along the first dimension
                perturbed_data = perturbed_data[permutation]

            return perturbed_data, self.labels[idx]
        
    
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
            'perturbations': self.perturbations,
            'include_original': self.include_original,
            'shuffle': self.shuffle,
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
            perturbations=checkpoint['perturbations'],
            include_original=checkpoint['include_original'],
            shuffle=checkpoint['shuffle'],
        )
    
def plot_perturbation_layers(dataset, idx):
    """Plots each layer of perturbation for a given instance from the REPDataset.

    Args:
        dataset (REPDataset): The REPDataset instance.
        idx (int): The index of the instance to plot.
    """
    # Get the perturbed data and label for the specified index
    perturbed_data, label = dataset[idx]
    
    # Get the number of perturbation layers
    num_layers = perturbed_data.size(0)
    
    # Create a grid of subplots
    num_cols = 4
    num_rows = (num_layers + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 3*num_rows))
    
    # Flatten the axes array for easier indexing
    axes = axes.flatten()
    
    # Plot each layer of perturbation
    for i in range(num_layers):
        layer_data = perturbed_data[i].squeeze().cpu().numpy()
        axes[i].imshow(layer_data, cmap='gray')
        axes[i].set_title(f"Layer {i+1}")
        axes[i].axis('off')
    
    # Remove any unused subplots
    for i in range(num_layers, len(axes)):
        fig.delaxes(axes[i])
    
    # Adjust the spacing between subplots
    plt.tight_layout()
    
    # Show the plot
    plt.show()