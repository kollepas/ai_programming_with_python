@echo on
python predict.py my_images/rose.jpg save_directory/checkpoint.pth --top_k 10 --category_names cat_to_name.json --gpu cpu > results_predict_model.txt
