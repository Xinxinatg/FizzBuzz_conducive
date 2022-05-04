import numpy as np
import pandas as pd
from tqdm import tqdm
def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])

NUM_DIGITS = 14
upperlimit=2**NUM_DIGITS
X=[]
y=[]
for num in tqdm(range(1, upperlimit)):
    X.append(np.array(binary_encode(num, NUM_DIGITS)))
    string = ""
    if num % 3 == 0:
       string = string + "Fizz"
    if num % 5 == 0:
       string = string + "Buzz"
    if num % 5 != 0 and num % 3 != 0:
        string = string + str(num)
    if string=="Fizz":
         y.append(np.array(0))
      #  y.append(np.array([0, 0, 0, 1]))
    elif string=="Buzz":
         y.append(np.array(1))
      #  y.append(np.array([0, 0, 1, 0]))
    elif string=="FizzBuzz":
         y.append(np.array(2))
      #  y.append(np.array([0, 1, 0, 0]))
    else:
         y.append(np.array(3))
      #  y.append(np.array([1, 0, 0, 0]))
    # print(string)


# X=pd.DataFrame(X)
X=np.array(X)
y=np.array(y)
# y=ravel(pd.DataFrame(y))
# y=pd.DataFrame(y)

X_train, X_val,X_test=X[int(len(X)*0.3):],X[int(len(X)*0.2):int(len(X)*0.3)],X[:int(len(X)*0.2)]
y_train, y_val,y_test=y[int(len(X)*0.3):],y[int(len(X)*0.2):int(len(X)*0.3)],y[:int(len(X)*0.2)]
X_train_pd=pd.DataFrame(X_train)
y_train_pd=pd.DataFrame(X_train)
X_val_pd=pd.DataFrame(X_val)
y_val_pd=pd.DataFrame(y_val)
X_test_pd=pd.DataFrame(X_test)
y_test_pd=pd.DataFrame(y_test)
X_train_pd.to_csv('X_train.csv',index=False)
y_train_pd.to_csv('y_train.csv',index=False)
X_val_pd.to_csv('X_val.csv',index=False)
y_val_pd.to_csv('y_val.csv',index=False)
X_test_pd.to_csv('X_test.csv',index=False)
y_test_pd.to_csv('y_test.csv',index=False)
