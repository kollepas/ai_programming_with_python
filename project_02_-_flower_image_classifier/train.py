# PROGRAMMER: Pascal Kolleth
# DATE CREATED: 02.01.2019
# PURPOSE: Train an image classifier network on a dataset using a pretrained model.
#          Save the trained model as a checkpoint so that it can be loaded later on.
# INPUTS: The program takes eight input arguments from the command line:
#     01. <data_dir> (mandatory): directory of image data
#     02. <save_dir> (optional / default=save_directory): directory to save checkpoints
#     03. <arch> (optional / default=vgg16): architecture model name
#     04. <learning_rate> (optional / default=0.001): learning rate in decimal
#     05. <hidden_units> (optional / default=512): integer number of hidden units
#     06. <output_units> (optional / default=102): integer number of output units
#     07. <epochs> (optional / default=4): integer number of epochs
#     08. <gpu> (optional / default=cpu): use GPU for training (no input argument --> cuda)
# OUTPUTS: The program saves a trained model in the <save_dir> under the name <checkpoint.pth>

# Import all necessary python modules
from torch import nn
from torch import optim
from torchvision import models
from model_fun import build_classifier
from model_fun import deep_learning
from model_fun import data_transforms as dt
from model_fun import dataloaders as dl
from model_fun import save_checkpoint
from utility_fun import parse_input_arguments as pia

# Parse the input arguments from the command line
inp_args = pia('train')

# Define the transforms for the training and validation sets
train_data_transforms = dt(45, None, 224, None, True, True)
valid_data_transforms = dt(None, 255, 224, True, True, True)

# Load the datasets and define the dataloaders for the training and validation sets
train_dataloader, train_dataset = dl(inp_args.data_dir + '/train', train_data_transforms, shuffle=True)
valid_dataloader, _ = dl(inp_args.data_dir + '/valid', valid_data_transforms)

# Load the pre-trained network as defined by the user
model = eval('models.' + inp_args.arch + '(pretrained=True)')

# Freeze the model parameters
for p in model.parameters():
    p.requires_grad = False

# Build a new classifier as defined by the user
classifier = build_classifier(model.classifier[0].in_features, inp_args.hidden_units, inp_args.output_units)

# Replace the model classifier with the newly created classifier
model.classifier = classifier

# Set the criterion
criterion = nn.NLLLoss()

# Set the optimization algorithm
optimizer = optim.Adam(model.classifier.parameters(), lr=inp_args.learning_rate)

# Start to train the model
deep_learning(model, train_dataloader, valid_dataloader, inp_args.epochs, 25, criterion, optimizer, inp_args.gpu)

# Attach the <class_to_idx> mapping from one of the datasets to the model
# in order to save it along with the model
model.class_to_idx = train_dataset.class_to_idx

# Save the model
save_checkpoint(model, optimizer, inp_args.epochs, inp_args.save_dir + '/checkpoint')
