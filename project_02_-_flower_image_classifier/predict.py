# PROGRAMMER: Pascal Kolleth
# DATE CREATED: 02.01.2019
# PURPOSE: Predict the top-K probabilities and names of an image using a pretrained network (model)
#          that is loaded from a checkpoint file.
# INPUTS: The program takes five input arguments from the command line:
#     01. <input> (mandatory): Full path of a single image
#     02. <checkpoint> (mandatory): Full path of a checkpoint file
#     03. <top_k> (optional / default=5): Number of top-K results
#     04. <category_names> (optional / default='cat_to_name.json'): Full path of a JSON object (file)
#                                                                   that contains category to name mapping
#     05. <gpu> (optional / default=cpu): use GPU for prediction (no input argument --> cuda)
# OUTPUTS: The program prints the resulting probabilities and names in the command window

# Import all necessary python modules
import torch
from model_fun import load_checkpoint
from utility_fun import cat2name
from utility_fun import parse_input_arguments as pia
from utility_fun import process_image

# Parse the input arguments from the command line
inp_args = pia('predict')

# Load the model, optimizer and number of epochs from the checkpoint file
model, optimizer, epochs = load_checkpoint(inp_args.checkpoint, inp_args.gpu)

# Create a full dictionary in order to decode category names
cat_to_name = cat2name(inp_args.category_names)
full_dict = dict()
for keys, values in model.class_to_idx.items():
    full_dict.update({keys: [values, cat_to_name[keys]]})

# Append the full dictionary from above to the model.
model.full_dict = full_dict

# Put the model into evaluation mode and move it to the appropriate device
model.eval()
model.to(inp_args.gpu)

# Import the image in Numpy array format
img = process_image(inp_args.input)

# Convert the Numpy image into a tensor on the appropriate device
img_tensor = torch.from_numpy(img).float().to(inp_args.gpu)

# Resize the tensor for batching
img_tensor.unsqueeze_(0)

# Forward pass
output = model.forward(img_tensor)

# Calculate the probability (exponential is used, because the output is a LogSoftmax)
probability = torch.exp(output)

# Get the top-K probabilites along with the indices
probs, indices = torch.topk(probability, inp_args.top_k)

# Move the results back to CPU in order to use Numpy and convert it to a list
probs = probs.to('cpu').detach().numpy()[0]
indices = indices.to('cpu').detach().numpy()[0]

# Invert the attached full dictionary so that the indices are the keys
full_dict_inverted = dict([[values[0], [keys, values[1]]] for keys, values in model.full_dict.items()])

# Get the classes out of the inverted full dictionary
classes = [full_dict_inverted[i][0] for i in indices]
names = [full_dict[i][1] for i in classes]

# Print the results (top-K probabilites along with the names)
print('{:>20} ¦ {:>11}\n'.format('Flower Name', 'Probability'))
for i in range(0, len(probs)):
    print('{:>20} ¦ {:11.2%}'.format(names[i], probs[i]))
