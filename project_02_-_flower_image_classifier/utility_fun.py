# PROGRAMMER: Pascal Kolleth
# DATE CREATED: 01.01.2019
# PURPOSE: The script contains several functions that support <train.py> and <predict.py>
#          and are not directly related to the deep learning process. Particularly, the
#          functions support processing the inputs and outputs.

# Import all necessary python modules
import argparse
import json
import numpy as np
from PIL import Image

def parse_input_arguments(caller):
    """
    DESCRIPTION: Retrieve and parse the command line arguments depending on the
    caller of the function when <train.py> or <predict.py> is executed in a
    terminal window. If the user does not provide any input for the optional
    arguments, some default values will be used. The function returns a data
    structure that contains the command line argument.
    INPUT: <caller>: A string that depending on it's value ('train' or 'predict')
           parses the command line for different arguments
    OUTPUT: parser.parse_args(): data structure of parsed input arguments
    """
    #01. Create the parser
    parser = argparse.ArgumentParser()

    #02. Set up the command line arguments as required in the project definition
    if caller == 'train':
        parser.add_argument('data_dir', type=str,
                            help='directory of image data')
        parser.add_argument('--save_dir', type=str, default='save_directory',
                            help='directory to save checkpoints')
        parser.add_argument('--arch', type=str, default='vgg16',
                            help='architecture model name')
        parser.add_argument('--learning_rate', type=float, default=0.001,
                            help='learning rate in decimal')
        parser.add_argument('--hidden_units', type=int, default=512,
                            help='integer number of hidden units')
        parser.add_argument('--output_units', type=int, default=102,
                            help='integer number of output units')
        parser.add_argument('--epochs', type=int, default=4,
                            help='integer number of epochs')
        parser.add_argument('--gpu', nargs='?', const='cuda', default='cpu',
                            help='use GPU for training')
    elif caller == 'predict':
        parser.add_argument('input', type=str,
                            help='single image full path')
        parser.add_argument('checkpoint', type=str,
                            help='checkpoint full path')
        parser.add_argument('--top_k', type=int, default=5,
                            help='integer number of K most likely classes')
        parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                            help='file used for mapping of categories to real names')
        parser.add_argument('--gpu', nargs='?', const='cuda', default='cpu',
                            help='use GPU for inference')

    #03. Return the parsed arguments as the result
    return parser.parse_args()

def cat2name(category_names):
    """
    DESCRIPTION: The function loads a mapping from category label to category name.
    It expects the input to be a JSON object which will be read using the json module.
    The function returns a dictionary of mapped category labels and names.
    INPUT: <category_names>: A JSON object (file) that contains category label to
           category name mapping
    OUTPUT: <cat_to_name>: Dictionary of mapped category labels and names.
    """
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def process_image(image, scale_size=256, crop_size=224, normalize=True,
                  mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    DESCRIPTION: The function scales, crops, and normalizes a PIL image for a
    PyTorch model. The function returns a Numpy array that contains the image data
    INPUT: <image>: An image file (i.e. sample.jpg) to process
           <scale_size> (default=256): scale size of the image along the shorter edge
           <crop_size> (default=224): crop size in the image center
           <normalize> (default=True): Set to True will normalize the image
           <mean> (default=[0.485, 0.456, 0.406]): Mean used for normalization
           <std> (default=[0.229, 0.224, 0.225]): Standard deviation used for normalization
    OUTPUT: <np_img>: Numpy array that contains the image data
    """
    #01. Load image
    img = Image.open(image)

    #02. Resize image, keeping the aspect ratio
    width, height = img.size
    if width > height:
        img = img.resize((int(width * (crop_size / height)), crop_size))
    else:
        img = img.resize((crop_size, int(height * (crop_size / width))))

    #03. Crop image in the center to the required size
    img = img.crop(((img.size[0] - crop_size)/2, (img.size[1] - crop_size)/2,
                    (img.size[0]+ crop_size)/2, (img.size[1] + crop_size)/2))

    #04. Convert PIL image to Numpy array
    np_img = np.array(img)

    #05. Normalize the image
    if normalize:
        np_img = ((np_img/255) - mean) / std

    #06. Reorder dimensions as PyTorch expects the color channel to be the first
    #    dimension but it's the third dimension in the PIL image and Numpy array
    np_img = np_img.transpose(2, 0, 1)

    #07. Return the Numpy array image as the result
    return np_img
