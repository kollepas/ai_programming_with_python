@echo on
python check_images.py --dir your_images/ --arch resnet  --dogfile dognames.txt > your_images_results_resnet.txt
python check_images.py --dir your_images/ --arch alexnet --dogfile dognames.txt > your_images_results_alexnet.txt
python check_images.py --dir your_images/ --arch vgg  --dogfile dognames.txt > your_images_results_vgg.txt
