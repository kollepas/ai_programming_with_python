# Pet Image Classifier
---
> Load a pretrained Neutral Network model of choise to classify pet images and identify dog breeds.

## Overview
---
In the **Pet Image Classifier** project, a pretrained Convolutional Neutral Network of choise (resnet18, alexnet or vgg16) is used to classify pet images located in a directory (e.g. [your_images](https://github.com/kollepas/ai_programming_with_python/tree/master/project_01_-_pet_image_classifier/your_images)). The classification results will be written and save in a text file.

## Installation
---
- Clone/Download the folder.
- Beside the basic 'standard' Python packages, `torch` and `torchvision` must be installed.

## How to use?
---
- Put some pet images into the [your_images](https://github.com/kollepas/ai_programming_with_python/tree/master/project_01_-_pet_image_classifier/your_images) folder.
- Make sure that the file name includes the dog's name if you want to verify it (similar to my example [Basset_hound_01.jpg](https://github.com/kollepas/ai_programming_with_python/blob/master/project_01_-_pet_image_classifier/your_images/Basset_hound_01.jpg)). The name must match an entry in the file [dognames.txt](https://github.com/kollepas/ai_programming_with_python/blob/master/project_01_-_pet_image_classifier/dognames.txt); but underscores as a separator and capital letter at the beginning are OK. Of course, if you do not know the dogs breed, call the file as you like and the result will list it under _Misclassified Breed's of Dog_ stating what it suggests.
- Run the batch file [run_models_batch_on_your_images.bat](https://github.com/kollepas/ai_programming_with_python/blob/master/project_01_-_pet_image_classifier/run_models_batch_on_your_images.bat) and let all three CNN models classify the images in the [your_images](https://github.com/kollepas/ai_programming_with_python/tree/master/project_01_-_pet_image_classifier/your_images) folder.
- The batch will create a result text file for each model.
- Scroll to the end of the text file to find the summary of how many images has been correctly classified as dog vs. non-dog and how many of the dog images have correctly classified breeds. For misclassified dog breeds, the text file also states what the pet images breed is expected to be from the file name ('Pet Image:) vs. what the classifier guessed ('Classifier Labels:).

![Sample Result](https://github.com/kollepas/ai_programming_with_python/blob/master/project_01_-_pet_image_classifier/graphics/sample_result.JPG)

So in my example, the vgg model correctly recognized that 3 images are of dogs and 3 are of anything else but dogs. Furthermore, it correctly identified the Basset_hound_01 and Basset_hound_02 as a basset hound but thought that the lazy black russian terrier (in fact I am not even sure if it is a purebred terrier as he used to be a stray dog :grin::dog::question:) is a miniature poodle.

## License
---
This project is licensed under the MIT License - see the [LICENSE](https://github.com/kollepas/ai_for_trading/blob/master/LICENSE) file for details.
