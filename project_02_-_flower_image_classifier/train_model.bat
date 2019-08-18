@echo on
python train.py flowers/ --save_dir save_directory/ --arch vgg16  --learning_rate 0.001 --hidden_units 512 --output_units 102 --epochs 1 --gpu cpu > results_train_model.txt
