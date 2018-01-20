------------------------------------
How to execute
------------------------------------
Note: Use python 2.7 version
----------------------------
1. Preprocessing:
-----------------
Run Preprocessing.py using command with filenames as argument:
python Preprocessing.py input_file_path output_file_path

Eg:
python Preprocessing.py D:\UTD\ML\Assignment3\Iris_data.txt D:\UTD\ML\Assignment3\input_prep.csv
------------------------------------

2. Building Neural Network:
----------------------------
Run Neural.py using command with corr arguments:
python Neural.py output_file_path training_percentage max_iterations hidden_layer num_neurons
Eg:
python Neural.py D:\UTD\ML\Assignment3\input_prep.csv 80 200 1 8
