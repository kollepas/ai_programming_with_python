# Flower Image Classifier
---
> Load a pretrained Neutral Network model and train the classifier (on images of flowers). Use the trained model to predict (classify) new flower images.

## Overview
---
In the **Flower Image Classifier** project, a pretrained Convolutional Neutral Network (vgg16) is used and the classifier layer is trained on images of flowers. The training, validation and testing data is located in the split _flowers.rar_ file that will have to be unpacked. It is also shown how a trained model can be saved, loaded and used to predict (classify) new flower images.

## Installation
---
- Clone/Download the folder.
- Unrar the _flowers_ archive that is split into 4 parts to one single folder.
- Beside the basic 'standard' Python packages, `torch` and `torchvision` must be installed.

## How to use?
---
The project actually consist of two part. First, everything had to be built in the Jupyter Notebook and second, move it to standalone .py files so that it can be run from the command prompt:

### Part 1: Jupyter Notebook:
Open the [Jupyter Notebook file](https://github.com/kollepas/ai_programming_with_python/blob/master/project_02_-_flower_image_classifier/project_02_-_flower_image_classifier.ipynb) and run through the code blocks. The notebook itself comes with detailed explanation about each step.


### Part 2: Command Line:
- Put some flower images into the [my_images](https://github.com/kollepas/ai_programming_with_python/tree/master/project_02_-_flower_image_classifier/my_images) folder.
- Initially, the model will have to be trained. In order to do that, `train.py` must be run in the command prompt. `train.py` takes 8 input arguments, whereof the first one is mandatory:
```python
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
```
- So, as an example, the following command line will start to train the model and save a _checkpoint.pth_ file in the 'save_directory':
```
python train.py flowers/ --save_dir save_directory/ --arch vgg16  --learning_rate 0.001 --hidden_units 512 --output_units 102 --epochs 1 --gpu cpu > results_train_model.txt
```
- After the model is trained, run the `predict.py` in the command prompt. `predict.py` takes 5 input arguments, whereof the first two are mandatory:
```python
# INPUTS: The program takes five input arguments from the command line:
#     01. <input> (mandatory): Full path of a single image
#     02. <checkpoint> (mandatory): Full path of a checkpoint file
#     03. <top_k> (optional / default=5): Number of top-K results
#     04. <category_names> (optional / default='cat_to_name.json'): Full path of a JSON object (file)
#                                                                   that contains category to name mapping
#     05. <gpu> (optional / default=cpu): use GPU for prediction (no input argument --> cuda)
# OUTPUTS: The program prints the resulting probabilities and names in the command window
```
- So, as an example, the following command line will load the model _checkpoint.pth_ located in the 'save_directory' folder and do a prediction of my input image of [rose.jpg](https://github.com/kollepas/ai_programming_with_python/blob/master/project_02_-_flower_image_classifier/my_images/rose.jpg):
```
python predict.py my_images/rose.jpg save_directory/checkpoint.pth --top_k 10 --category_names cat_to_name.json --gpu cpu > results_predict_model.txt
```
- As a result, the model will show the `top_k` most probable categories along with the probability value.

![Sample Result](https://github.com/kollepas/ai_programming_with_python/blob/master/project_02_-_flower_image_classifier/assets/sample_result.JPG)

As the demo shows here, even after training the model for only one epoch, the Neural Network is pretty sure (probability of 83.37%) that my input image [rose.jpg](https://github.com/kollepas/ai_programming_with_python/blob/master/project_02_-_flower_image_classifier/my_images/rose.jpg) is indeed a :rose:

For convenience, two batch files for [training](https://github.com/kollepas/ai_programming_with_python/blob/master/project_02_-_flower_image_classifier/train_model.bat) and [predicting](https://github.com/kollepas/ai_programming_with_python/blob/master/project_02_-_flower_image_classifier/predict_model.bat) are also included according to the example above that operate on the local directory.

## License
---
This project is licensed under the MIT License - see the [LICENSE](https://github.com/kollepas/ai_for_trading/blob/master/LICENSE) file for details.
