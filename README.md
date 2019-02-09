# Lottery

## Dependency

This project is dependent on Python 

- keras == 2.2.0
- numpy == 1.14.0
- pandas == 0.22.0
- matplotlib == 2.1.2
- scikit-learn == 0.20
- argparse == 3.2

## Usage



### Training & Prediction(testing)

To train a model, do:

```shell
# There are up to 5 parameters: mode, data path, output folder, batch size, patience :
python3 main.py --mode train --source_data_path 'your csv' --name example --batch_size 120 --patience 50
```

To predict, do:

```shell
# There are up to 5 parameters: mode, data path, output folder, batch size, patience :
python3 main.py --mode test --source_data_path 'your csv' --name example --batch_size 120 --patience 50
```
