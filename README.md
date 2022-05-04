# FizzBuzz_conducive



## Code details

* Users can run the model_inference.py to identify the RBPs. 

* generate_embeddings.py is implemented for emebeddings generation. 
* train.py is implemented for re-training the customized deep learning model from scratch. 



## Dependency
* python 3.8
* torch 1.7.1
* cuda 11.0
* scikit_learn 0.22.2 



## Installation Guide

*  Install from Github 
```python
git clone https://github.com/Xinxinatg/FizzBuzz_conducive
cd FizzBuzz_conducive
pip install -r requirements.txt
```
## Steps for re-training the model for FizzBuzz prediction: 
- Run the code 
    - Generate the dataset for training, validation and testing, there will be X_train.csv, y_train.csv, X_val.csv, y_val.csv, X_test.csv and y_test.csv files afterwards.
    ```
    python generate_data.py.
    ```
    - Re-train the model: the trained model will be store in the Model folder:
    ```
    python train.py.
    ```

## Predicting FizzBuzz using trained models:
- Run the code, please enter a positive integer after the prommt, the positive integer has to be smaller than 16384.
    ```
    python FizzBuzz.py
    ```

