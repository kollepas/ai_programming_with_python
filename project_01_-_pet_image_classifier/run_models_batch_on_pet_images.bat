@echo on
python check_images.py --dir pet_images/ --arch resnet  --dogfile dognames.txt > pet_images_results_resnet.txt
python check_images.py --dir pet_images/ --arch alexnet --dogfile dognames.txt > pet_images_results_alexnet.txt
python check_images.py --dir pet_images/ --arch vgg  --dogfile dognames.txt > pet_images_results_vgg.txt
