#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/adjust_results4_isadog.py
#
# PROGRAMMER: Pascal Kolleth
# DATE CREATED: 18.11.2018
# REVISED DATE:
# PURPOSE: Create a function adjust_results4_isadog that adjusts the results
#          dictionary to indicate whether or not the pet image label is of-a-dog,
#          and to indicate whether or not the classifier image label is of-a-dog.
#          All dog labels from both the pet images and the classifier function
#          will be found in the dognames.txt file. We recommend reading all the
#          dog names in dognames.txt into a dictionary where the 'key' is the
#          dog name (from dognames.txt) and the 'value' is one. If a label is
#          found to exist within this dictionary of dog names then the label
#          is of-a-dog, otherwise the label isn't of a dog. Alternatively one
#          could also read all the dog names into a list and then if the label
#          is found to exist within this list - the label is of-a-dog, otherwise
#          the label isn't of a dog.
#         This function inputs:
#            -The results dictionary as results_dic within adjust_results4_isadog
#             function and results for the function call within main.
#            -The text file with dog names as dogfile within adjust_results4_isadog
#             function and in_arg.dogfile for the function call within main.
#           This function uses the extend function to add items to the list
#           that's the 'value' of the results dictionary. You will be adding the
#           whether or not the pet image label is of-a-dog as the item at index
#           3 of the list and whether or not the classifier label is of-a-dog as
#           the item at index 4 of the list. Note we recommend setting the values
#           at indices 3 & 4 to 1 when the label is of-a-dog and to 0 when the
#           label isn't a dog.
#
##
# TODO 4: Define adjust_results4_isadog function below, specifically replace the None
#       below by the function definition of the adjust_results4_isadog function.
#       Notice that this function doesn't return anything because the
#       results_dic dictionary that is passed into the function is a mutable
#       data type so no return is needed.
#
# from get_pet_labels import get_pet_labels
# from classify_images import classify_images
def adjust_results4_isadog(results_dic, dogfile):
    """
    Adjusts the results dictionary to determine if classifier correctly
    classified images 'as a dog' or 'not a dog' especially when not a match.
    Demonstrates if model architecture correctly classifies dog images even if
    it gets dog breed wrong (not a match).
    Parameters:
      results_dic - Dictionary with 'key' as image filename and 'value' as a
                    List. Where the list will contain the following items:
                  index 0 = pet image label (string)
                  index 1 = classifier label (string)
                  index 2 = 1/0 (int)  where 1 = match between pet image
                    and classifer labels and 0 = no match between labels
                ------ where index 3 & index 4 are added by this function -----
                 NEW - index 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and
                            0 = pet Image 'is-NOT-a' dog.
                 NEW - index 4 = 1/0 (int)  where 1 = Classifier classifies image
                            'as-a' dog and 0 = Classifier classifies image
                            'as-NOT-a' dog.
     dogfile - A text file that contains names of all dogs from the classifier
               function and dog names from the pet image files. This file has
               one dog name per line dog names are all in lowercase with
               spaces separating the distinct words of the dog name. Dog names
               from the classifier function can be a string of dog names separated
               by commas when a particular breed of dog has multiple dog names
               associated with that breed (ex. maltese dog, maltese terrier,
               maltese) (string - indicates text file's filename)
    Returns:
           None - results_dic is mutable data type so no return needed.
    """
    # Import dogfile as list and clean it
    with open(dogfile) as file:
        dognames = file.read().splitlines()
    # Create dognames dictionary
    dognames_dic = dict()
    for breed in dognames:
        dogs = [x.strip() for x in breed.split(',')]
        for dog in dogs:
            if dog not in dognames_dic:
                # If the dogname is new, add it to the dictionary with default
                # value = 1 ...
                dognames_dic[dog] = 1
            else:
                # ... Else print a warning that the key already exists
                print(('Warning: Key {} already exists in dognames_dic with '
                       'value = {}').format(dog, dognames_dic[dog]))
    # Loop through results_dic directory items one-by-one
    for key, value in results_dic.items():
        # Extend dictionary for pet image labels in dognames_dic 1 (if existing)
        # 0 (in not existing)
        results_dic[key].extend([dognames_dic.get(value[0], 0)])
        # Extend dictionary for any classifier function label in dognames_dic
        # 1 (if existing) or 0 (in not existing)
        model_labels = value[1].split(',')
        for i, item in enumerate(model_labels):
            model_labels[i] = (item.strip()).lower()
        if any(item in dognames_dic for item in model_labels):
            results_dic[key].extend([1])
        else:
            results_dic[key].extend([0])

# Stand-alone testing
# image_dir = 'C:/Users/Csiga/Documents/Python Scripts/project1_classify_pet_images_pascal/pet_images/'
# results_dic = get_pet_labels(image_dir)
# classify_images(image_dir, results_dic, 'resnet')
# print(results_dic)
# adjust_results4_isadog(results_dic, 'dognames.txt')
# print(results_dic)
