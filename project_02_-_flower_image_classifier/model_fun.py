# PROGRAMMER: Pascal Kolleth
# DATE CREATED: 01.01.2019
# PURPOSE: The script contains several functions that support <train.py> and <predict.py>
#          and are directly related to the deep learning process. Particularly, any function
#          that is related to a torchvision model, training or valuation will be defined here.

# Import all necessary python modules
from collections import OrderedDict
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def data_transforms(rotation=None, resize=None, crop_size=224, center_crop=False, to_tensor=True,
                    normalize=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    DESCRIPTION: The function composes a list of transformations (rotation, scale, crop,
    normalization and transformation to a tensor).
    INPUT: <rotation> (default=None): Angle of random rotation
           <resize> (default=None): Resize of an image
           <crop_size> (default=224): crop size on an image
           <center_crop> (default=False): Set to True will crop the image in the center.
                                          Set to False will do a random resize crop.
           <to_tensor> (default=True): Transform the image to a tensor with range [0, 1]
           <normalize> (default=True): Set to True will normalize the image
           <mean> (default=[0.485, 0.456, 0.406]): Mean used for normalization
           <std> (default=[0.229, 0.224, 0.225]): Standard deviation used for normalization
    OUTPUT: transforms.Compose(trans_list): A transform of composed transformations
    """
    #01. Create a list <trans_list> that contains all transforms as defined by the input arguments
    trans_list = []
    if rotation is not None:
        trans_list.append(transforms.RandomRotation(rotation))
    if resize is not None:
        trans_list.append(transforms.Resize(resize))
    if crop_size is not None:
        if center_crop:
            trans_list.append(transforms.CenterCrop(crop_size))
        else:
            trans_list.append(transforms.RandomResizedCrop(crop_size))
    if to_tensor:
        trans_list.append(transforms.ToTensor())
    if normalize:
        trans_list.append(transforms.Normalize(mean=mean, std=std))

    #02. Return the composed transform as a result
    return transforms.Compose(trans_list)

def dataloaders(img_dir, transform, batch_size=64, shuffle=False):
    """
    DESCRIPTION: The function loads a dataset with ImageFolder using the
    transforms from the input argument and defines the dataloader.
    INPUT: <img_dir>: Directory of images
           <transforms>: A composed list of transforms
           <batch_size> (default=64): Batch size how many samples per batch to load
           <shuffle> (default=False): Set to True will reshuffle the data at every epoch
    OUTPUT: <dataset>: Dataset of images where transform was applied
            <dataloader>: Dataloader defined from the dataset
    """
    dataset = datasets.ImageFolder(img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader, dataset

def build_classifier(input_size, hidden_size, output_size, p_drop=0.5):
    """
    DESCRIPTION: The function builds a new model classifier with input-
    hidden- and output size defined by the input arguments. It will also
    include dropout using the dropout rate from the input argument. The
    output will be defined as a LogSoftmax. The function returns the newly
    defined classifier.
    INPUT: <input_size>: The size of the input layer
           <hidden_size>: The size of the hidden layer
           <output_size>: The size of the output layer. Typically the number
                          categories of whatever will be classified
           <p_drop> (default=0.5): Dropout probability
    OUTPUT: <classifier>: New classifier that can be attached to a model
    """
    classifier = nn.Sequential(OrderedDict([
                               ('input_layer',nn.Linear(input_size, hidden_size * 2)),
                               ('relu1',nn.ReLU()),
                               ('drop1',nn.Dropout(p=p_drop)),
                               ('hidden_layer1',nn.Linear(hidden_size * 2, hidden_size)),
                               ('relu2',nn.ReLU()),
                               ('drop2',nn.Dropout(p=p_drop)),
                               ('output_layer',nn.Linear(hidden_size, output_size)),
                               ('softmax',nn.LogSoftmax(dim=1))]))

    return classifier

def deep_learning(model, train_dataloader, valid_dataloader, epochs, print_interval, criterion, optimizer, device='cpu'):
    """
    DESCRIPTION: The function trains a model using the training data from <train_dataloader>.
    It also validates the results using <valid_dataloader> every <print_interval> and displays
    the results.
    INPUT: <model>: A torchvision model that can be trained
           <train_dataloader>: Dataloader of training data
           <valid_dataloader>: Dataloader of validation data
           <epochs>: Number of epochs to loop through
           <print_interval>: The interval in which the results are printed
           <criterion>: A defined criterion for loss calculation
           <optimizer>: A defined optimization algorithm
           <device> (default='cpu'): The device on which the calculation should be performed.
                                     Typically 'cuda' to make use of the GPU (fast) or 'cpu' (slow)
    """
    model.train()
    step = 0

    #01. Loop through the number of epochs
    for e in range(epochs):

        #02. Reset the running loss calculation
        running_loss = 0

        #03. Loop through the training data
        for images, labels in train_dataloader:
            step += 1

            #04. Move the environment to the specified device
            model.to(device)
            images = images.to(device)
            labels = labels.to(device)

            #05. Make sure to zero out the gradients to avoid accumulation
            optimizer.zero_grad()

            #06. Forward pass
            outputs = model.forward(images)

            #07. Loss calculation
            loss = criterion(outputs, labels)

            #08. Backpropagation
            loss.backward()

            #09. Parameter update
            optimizer.step()

            #10. Update the running loss
            running_loss += loss.item()

            #11. Display the running loss if it's a multiple of the print interval
            if step % print_interval == 0:

                #12. Put the model into evaluation mode
                model.eval()

                #13. Disable gradients for the validation to save memory
                with torch.no_grad():
                    valid_loss, valid_accuracy = validation(model, valid_dataloader, criterion, device)

                #14. Print the validation results
                print('Epoch: {}/{} ¦ '.format(e+1, epochs),
                      'Training Loss: {:.4f} ¦ '.format(running_loss/print_interval),
                      'Validation Loss: {:.4f} ¦ '.format(valid_loss/len(valid_dataloader)),
                      'Validation Accuracy: {:.2%}'.format(valid_accuracy/len(valid_dataloader)))

                running_loss = 0

                #15. Put the model back into training mode
                model.train()

def validation(model, dataloader, criterion, device='cpu'):
    """
    DESCRIPTION: The function validates a model using the dataloader (typically a
    validation set) and returns the accuracy of the model in percentage along with the loss
    INPUT: <model>: A torchvision model that can be validated
           <dataloader>: Dataloader of validation data
           <criterion>: A defined criterion for loss calculation
           <device> (default='cpu'): The device on which the calculation should be performed.
                                     Typically 'cuda' to make use of the GPU (fast) or 'cpu' (slow)
    OUTPUT: <loss>: The loss of the error term
            <accuracy>: The model accuracy in percentage
    """
    loss = 0
    accuracy = 0

    #01. Loop through the dataloader (i.e. validation or testing)
    for images, labels in dataloader:

        #02. Move the environment to the specified device
        model.to(device)
        images = images.to(device)
        labels = labels.to(device)

        #03. Forward pass
        output = model.forward(images)

        #04. Loss calculation (only the loss item itself because backpropagation is not needed here)
        loss += criterion(output, labels).item()

        #05. Calculate the probability (exponential is used, because the output is a LogSoftmax)
        probability = torch.exp(output)

        #06. Check where the probablity output matches the input labels
        equality = (labels.data == probability.max(dim=1)[1])

        #07. Calculate accuracy (equality returns a list of ones and zeros; so the mean is the accuracy)
        accuracy += equality.type(torch.cuda.FloatTensor if device=='cuda' else torch.FloatTensor).mean()

    #08. Return <loss> and <accuracy> as the result of the validation function
    return loss, accuracy

def save_checkpoint(model, optimizer, epochs, checkpoint_name):
    """
    DESCRIPTION: The function saves a torchvision model along with all attributes
    required to recreate it from scratch.
    INPUT: <model>: A torchvision model to save
           <optimizer>: optimization algorithm that was used
           <epochs>: Number of epochs used
           <checkpoint_name>: Name of the checkpoint file without the file extension
    """
    #01. Create a checkpoint dictionary <cp> that contains all information to recreate the model architecture
    cp = {'model': model,
          'model_state_dict': model.state_dict(),
          'class_to_idx': model.class_to_idx,
          'optimizer': optimizer,
          'optimizer_state_dict': optimizer.state_dict(),
          'epochs': epochs}

    #02. Save the checkpoint
    torch.save(cp, checkpoint_name + '.pth')

def load_checkpoint(checkpoint_name, device='cpu'):
    """
    DESCRIPTION: The function loads a torchvision model along with all attributes.
    INPUT: <checkpoint_name>: Full path of the checkpoint file (incl. file extension)
           <device> (default='cpu'): The device on which the model should be loaded.
                                     Typically 'cuda' to make use of the GPU (fast) or 'cpu' (slow)
    OUTPUT: <model>: The model along with its state_dict as it was saved in the checkpoint file
            <optimizer>: The optimizer along with its state_dict as it was saved in the checkpoint file
            <epochs>: The number of epochs as it was saved in the checkpoint file
    """
    #01. Load the checkpoint
    if device == 'cuda':
        device += ':0'
    cp = torch.load(checkpoint_name, map_location=device)

    #02. Apply the checkpoint attributes
    model = cp['model']
    model.load_state_dict(cp['model_state_dict'])
    model.class_to_idx = cp['class_to_idx']
    optimizer = cp['optimizer']
    optimizer.load_state_dict(cp['optimizer_state_dict'])
    epochs = cp['epochs']

    #03. Return the updated results
    return model, optimizer, epochs
