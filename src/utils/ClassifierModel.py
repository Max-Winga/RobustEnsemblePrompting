# model code for simple convolutional nerual network followed by fully connected layers
import numpy as np
from matplotlib import pyplot as plt
import time
import copy

import torch
import torch.nn as nn

class ClassifierModel(nn.Module):
    """A PyTorch Module for a convolutional neural network classifier.

    This class provides methods for training and testing a classifier model, as well as visualizing its performance.
    The model consists of some number of convolution layers followed by fully connected layers.

    Attributes:
        channel_widths (list of int): The number of channels for each convolutional layer.
        linear_sizes (list of int): The sizes of the fully connected layers.
        kernel (int): The size of the kernel for the convolutional layers.
        pooling (torch.nn.Module): The pooling layer.
        nonlinearity (torch.nn.Module): The nonlinearity used in the model. Defaults to nn.ReLU().
        num_classes (int): The number of classes to predict. Defaults to 2.
    """
    def __init__(self, channel_widths, linear_sizes, kernel, pooling, nonlinearity=nn.ReLU(), num_classes=2):
        """Initializes the ClassifierModel.

        Args:
            channel_widths (list of int): The number of channels for each convolutional layer.
                Should begin with size of number of frames (1 for standard data, 2 for ratiometric).
            linear_sizes (list of int): The sizes of the fully connected layers.
            kernel (int): The size of the kernel for the convolutional layers.
            pooling (torch.nn.Module): The pooling layer.
            nonlinearity (torch.nn.Module): The nonlinearity used in the model. Defaults to nn.ReLU().
            num_classes (int): The number of classes to predict. Defaults to 2.
        """
        super(ClassifierModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.channel_widths = channel_widths
        self.linear_sizes = linear_sizes
        self.kernel = kernel
        self.pooling = pooling
        self.nonlinearity = nonlinearity
        self.num_classes = num_classes
        
        layers = []
        for i in range(len(channel_widths)-2):
            layers.append(nn.Conv2d(channel_widths[i], channel_widths[i+1],
                                    kernel_size=kernel, padding=2, stride=1, bias=True))
            layers.append(nonlinearity)
        layers.append(nn.Conv2d(channel_widths[-2], channel_widths[-1],
                                    kernel_size=kernel, padding=2, stride=1, bias=True))
        self.backbone = nn.Sequential(*layers)
        self.global_pooling = pooling
        self.pool_size = pooling.output_size[0]*pooling.output_size[1]

        # Defining the fully connected layers
        fc_layers = []
        in_features = channel_widths[-1]*self.pool_size
        for size in linear_sizes:
            fc_layers.append(nn.Linear(in_features, size))
            fc_layers.append(nonlinearity)
            in_features = size
        self.fully_connected = nn.Sequential(*fc_layers)

        self.linear = nn.Linear(in_features, num_classes)  # score each class to obtain logits
        self.to(self.device)
        
        # Initialize lists to store performance metrics
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.best_val_accuracy = 0
        self.best_model_state_dict = None
        self.epochs_trained = 0
        self.training_parameter_history = []
        self.training_time = 0
        
    def forward(self, x):
        """Performs a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = x.to(self.device)
        B = x.size(0)
        features = self.backbone(x)
        pooled_features = self.global_pooling(features)
        pooled_features = pooled_features.view(B, -1)
        fc_output = self.fully_connected(pooled_features)
        logits = self.linear(fc_output)
        return logits
    
    def record_metrics(self, train_loss, train_acc, val_loss, val_acc):
        """Records the model's performance metrics.

        Args:
            train_loss (float): The training loss.
            train_acc (float): The training accuracy.
            val_loss (float): The validation loss.
            val_acc (float): The validation accuracy.
        """
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
    
    def validate(self, dataloader, criterion):
        """Validates the model on a given dataloader and criterion.

        Args:
            dataloader (DataLoader): The dataloader for validation data.
            criterion (torch.nn.Module): The loss function.

        Returns:
            tuple: A tuple containing the validation loss and accuracy.
        """
        val_loss = 0
        val_acc = 0
        # set model to eval mode (again, unnecessary here but good practice)
        self.eval()
        # don't compute gradients since we are not updating the model, saves a lot of computation
        with torch.no_grad():
            for images, targets in dataloader:
                images, targets = images.to(self.device), targets.to(self.device)
                class_logits = self(images)
                loss = criterion(class_logits, targets)
                val_loss += loss.item()
                val_acc += (class_logits.data.max(1)[1]).eq(targets).sum().item()
        return val_loss, val_acc

    def train_model(self, all_data, training_indices, validation_indices, config, verbose=True, printouts=20):
        """Trains the model on given data.

        Args:
            all_data (torch.utils.data.Dataset): The dataset containing all data.
            training_indices (array-like): The indices for the training data.
            validation_indices (array-like): The indices for the validation data.
            config (dict): The configuration for the training process, containing 'lr', 'n_epochs', and 'batch_size'.
            verbose (bool, optional): Whether to print progress during training. Defaults to True.
            printouts (int, optional): The number of times to print progress during training. Defaults to 20.
        """
        lr = config['lr'] # learning rate
        n_epochs = config['n_epochs'] # number of passes (epochs) through the training data
        batch_size = config['batch_size']
        self.training_parameter_history.append({'Epochs Scheduled': n_epochs, 'Learning Rate': lr, 'Batch Size': batch_size, 'Epochs Completed': 0, 
                                                'Training Indices': len(training_indices), 'Validation Indices': len(validation_indices), 'Training Time (s)': 0})
        if printouts > n_epochs: printouts = n_epochs
        previous_epochs = self.epochs_trained

        # set up optimizer and loss function
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # set up dataloaders
        train_sampler = torch.utils.data.SubsetRandomSampler(training_indices)
        val_sampler = torch.utils.data.SubsetRandomSampler(validation_indices)
        trainloader = torch.utils.data.DataLoader(all_data, batch_size=batch_size, sampler=train_sampler)
        valloader = torch.utils.data.DataLoader(all_data, batch_size=batch_size, sampler=val_sampler)
        
        try:
            for n in range(n_epochs):
                # set model to training mode (unnecessary for this model, but good practice)
                self.train()
                epoch_loss = 0
                epoch_acc = 0
                epoch_start = time.time()
                for images, targets in trainloader:
                    images, targets = images.to(self.device), targets.to(self.device)
                    optimizer.zero_grad() # zero out gradients
                    class_logits = self(images)
                    loss = criterion(class_logits, targets)
                    loss.backward() # backpropagate to compute gradients
                    optimizer.step() # update parameters using stochastic gradient descent
                    # update epoch statistics
                    epoch_loss += loss.item() # batch loss
                    epoch_acc += (class_logits.data.max(1)[1]).eq(targets).sum().item() # number of correct predictions
                    
                # validation
                epoch_loss /= len(trainloader)
                epoch_acc /= len(training_indices)
                val_loss, val_acc = self.validate(valloader, criterion)
                val_loss /= len(valloader)
                val_acc /= len(validation_indices)
                
                # log epoch information
                self.record_metrics(epoch_loss, epoch_acc, val_loss, val_acc)
                
                # save best model's state dict, if necessary
                if val_acc > self.best_val_accuracy:
                    self.best_val_accuracy = val_acc
                    self.best_model_state_dict = copy.deepcopy(self.state_dict())
                
                if verbose and (n+1) % (int(n_epochs/printouts)) == 0:
                    print('Epoch {}/{}: (Train) Loss = {:.4e}, Acc = {:.4f}, (Val) Loss = {:.4e}, Acc = {:.4f}'.format(
                        n + 1 + previous_epochs,
                        n_epochs + previous_epochs,
                        epoch_loss,
                        epoch_acc,
                        val_loss,
                        val_acc))
                self.epochs_trained += 1
                self.training_parameter_history[-1]['Epochs Completed'] += 1
                epoch_time = time.time() - epoch_start
                self.training_time += epoch_time
                self.training_parameter_history[-1]['Training Time (s)'] += epoch_time
        except KeyboardInterrupt:
            print("Training interrupted. Stopping after completing {} epochs of {} planned.".format(self.epochs_trained-previous_epochs, n_epochs))
        return

    def plot_model_results(self):
        """Plots the model's performance results."""
        plt.figure(figsize=(15, 10))
        plt.subplot(221)
        plt.semilogy(self.train_losses, color='royalblue')
        plt.xlabel('Epoch')
        plt.title('Training loss')
        plt.grid(True)
        plt.subplot(222)
        plt.plot(self.train_accs, color='darkorange')
        plt.xlabel('Epoch')
        plt.title('Training accuracy')
        plt.grid(True)
        plt.subplot(223)
        plt.plot(self.val_losses, color='royalblue')
        plt.xlabel('Epoch')
        plt.title('Validation loss')
        plt.grid(True)
        plt.subplot(224)
        plt.plot(self.val_accs, color='darkorange')
        plt.xlabel('Epoch')
        plt.title('Validation accuracy')
        plt.grid(True)
        plt.show()

    def get_training_time(self):
        """Gets the total training time for the model.

        Returns:
            float: The total training time in seconds.
        """
        self.training_time = 0
        for dict in self.training_parameter_history:
            self.training_time += dict['Training Time (s)']
        seconds = self.training_time % 60
        minutes = ((self.training_time-seconds) / 60) % 60
        hours = (((self.training_time-seconds) / 60) - minutes) / 60
        print(f"Model trained for: {hours} hrs, {minutes} mins, {seconds} s")
        return self.training_time
    
    def test_with_thresholds(self, model, dataset, thresholds=np.arange(0.5, 1, 0.01)):
        """Tests the model with various confidence thresholds.

        Args:
            model (torch.nn.Module): The model to test.
            dataset (torch.utils.data.Dataset): The dataset to test the model on.
            thresholds (array-like, optional): The confidence thresholds to test. Defaults to np.arange(0.5, 1, 0.01).

        Returns:
            list: A list of tuples, each containing a threshold, the test accuracy at that threshold, and the rejection ratio.
        """
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128)
        model.eval()
        
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in dataloader:
                images, targets = images.to(self.device), targets.to(self.device)
                class_probs = torch.nn.functional.softmax(model(images), dim=1)
                all_probs.extend(class_probs.tolist())
                all_targets.extend(targets.tolist())

        results = []
        for threshold in thresholds:
            correct_predictions = 0
            total_predictions = 0
            rejected_predictions = 0

            for i in range(len(all_probs)):
                max_prob = max(all_probs[i])
                predicted_class = all_probs[i].index(max_prob)
                if max_prob >= threshold:
                    total_predictions += 1
                    if predicted_class == all_targets[i]:
                        correct_predictions += 1
                else:
                    rejected_predictions += 1

            test_acc = correct_predictions / total_predictions if total_predictions > 0 else 0
            rejection_ratio = rejected_predictions / len(all_targets)
            results.append((threshold, test_acc, rejection_ratio))
        
        return results

    def plot_confidence_thresholding(self, datasets, thresholds=np.arange(0.5, 1, 0.01), use_best_model=False, colors=['orange', 'blue', 'green', 'red', 'black']):
        """Plots the model's accuracy and rejection ratio at various confidence thresholds.

        Args:
            datasets (list of tuple): A list of tuples, each containing a dataset name and dataset.
            thresholds (array-like, optional): The confidence thresholds to test. Defaults to np.arange(0.5, 1, 0.01).
            use_best_model (bool, optional): Whether to use the best model for testing. Defaults to False.
            colors (list of str, optional): The colors to use for plotting. Defaults to ['orange', 'blue', 'green', 'red', 'black'].
        """
        if use_best_model:
            model_state_dict = self.best_model_state_dict
        else:
            model_state_dict = self.state_dict()
        # load the state dict into a new instance of the model for testing
        model = ClassifierModel(self.channel_widths, self.linear_sizes, self.kernel, self.pooling, self.nonlinearity)
        model.load_state_dict(model_state_dict)
        model = model.to(self.device)

        plt.figure()
        for i in range(len(datasets)):
            dataset = datasets[i]
            results = np.array(self.test_with_thresholds(model, dataset[1], thresholds))
            plt.plot(results[:, 0], results[:, 1], color=colors[i], label = dataset[0] + ': Accuracy')
            plt.plot(results[:, 0], results[:, 2], color=colors[i], linestyle='--', label = dataset[0] + ': Rejection Ratio')
        plt.legend()
        ax = plt.gca()
        ax.set_yticks(np.arange(0, 1, 0.05), minor=True)
        ax.set_yticks(np.arange(0, 1.1, 0.1), minor=False)
        ax.grid(True, which='major')
        ax.grid(True, which='minor', ls='--')
        plt.ylabel("Accuracy/Rejection Ratios")
        plt.xlabel("Confidence Threshold")
        plt.title("Model Accuracy with Confidence Thresholding")
        plt.show()

    def confidence_histogram(self, dataset):
        """Plots a histogram of the model's confidence levels for each class on a given dataset.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to test the model on.
        """
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128)
        self.eval()

        confidences = [[] for _ in range(self.num_classes)]

        with torch.no_grad():
            for images, targets in dataloader:
                class_probs = torch.nn.functional.softmax(self(images), dim=1)

                for i in range(self.num_classes):
                    confidences[i].extend(class_probs[:, i].tolist())

        # Plot the histogram for each class
        for i in range(self.num_classes):
            plt.hist(confidences[i], bins=20, range=(0, 1), alpha=0.7, label='Class {}'.format(i))

        plt.title("Confidence Histogram")
        plt.xlabel("Confidence Score")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()

    def plot_classification_results(self, dataset, confidence_threshold=0.5):
        """Plots the model's classification results for each class on a given dataset.
        Shows correct/incorrect/rejected for each class at the given confidence threshold.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to test the model on.
            confidence_threshold (float, optional): The confidence threshold to use. Defaults to 0.5.
        """
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128)
        self.eval()

        class_counts = {i: {'correct': 0, 'incorrect': 0, 'rejected': 0} for i in range(self.num_classes)}

        with torch.no_grad():
            for images, targets in dataloader:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)

                for label in range(self.num_classes):
                    is_label = targets == label

                    confident_indices = softmax_outputs.max(1).values > confidence_threshold

                    correct_indices = predicted[is_label & confident_indices] == label
                    incorrect_indices = predicted[is_label & confident_indices] != label
                    rejected_indices = ~confident_indices & is_label
                    
                    class_counts[label]['correct'] += correct_indices.sum().item()
                    class_counts[label]['incorrect'] += incorrect_indices.sum().item()
                    class_counts[label]['rejected'] += rejected_indices.sum().item()

        labels = ['Class {}'.format(i) for i in range(self.num_classes)]
        categories = ['correct', 'incorrect', 'rejected']

        counts = [[class_counts[label][category] for category in categories] for label in range(self.num_classes)]

        counts = np.array(counts).T  # transpose so each row is a category
        
        x = np.arange(len(labels))  # label locations
        width = 0.2  # width of the bars

        fig, ax = plt.subplots()
        
        for i, category in enumerate(categories):
            ax.bar(x + i*width, counts[i], width, label=category)
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Counts')
        ax.set_title(f'Classification Results with Confidence Threshold: {confidence_threshold}')
        ax.set_xticks(x + width)
        ax.set_xticklabels(labels)
        ax.legend()

        plt.show()

    def save_model(self, PATH):
        """Saves the model to a file.

        Args:
            PATH (str): The path to save the model to.
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
            'best_val_accuracy': self.best_val_accuracy,
            'epochs_trained': self.epochs_trained,
            'training_parameter_history': self.training_parameter_history,
            'training_time': self.training_time,
            'channel_widths': self.channel_widths,
            'linear_sizes': self.linear_sizes,
            'kernel': self.kernel,
            'pooling': self.pooling,
            'nonlinearity': type(self.nonlinearity),  # save the type of nonlinearity
            'num_classes': self.num_classes,
            'best_model_state_dict': self.best_model_state_dict,
        }
        torch.save(checkpoint, PATH)

    @classmethod
    def load_model(cls, PATH):
        """Loads a model from a file.

        Args:
            PATH (str): The path to load the model from.

        Returns:
            ClassifierModel: The loaded model.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(PATH, map_location=torch.device(device))
        model = cls(
            channel_widths=checkpoint['channel_widths'],
            linear_sizes=checkpoint['linear_sizes'],
            kernel=checkpoint['kernel'],
            pooling=checkpoint['pooling'],
            nonlinearity=checkpoint['nonlinearity'](),  # instantiate the nonlinearity
            num_classes=checkpoint['num_classes'],  # load the number of classes
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.train_losses = checkpoint['train_losses']
        model.train_accs = checkpoint['train_accs']
        model.val_losses = checkpoint['val_losses']
        model.val_accs = checkpoint['val_accs']
        model.best_val_accuracy = checkpoint['best_val_accuracy']
        model.epochs_trained = checkpoint['epochs_trained']
        model.training_parameter_history = checkpoint['training_parameter_history']
        model.training_time = checkpoint['training_time']
        model.best_model_state_dict = checkpoint['best_model_state_dict']

        model.to(model.device)
        return model
