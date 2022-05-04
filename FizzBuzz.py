import torch.nn as nn
import numpy as np
import torch
import sys
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
NUM_DIGITS=14
model = torch.nn.Sequential(
    torch.nn.Linear(NUM_DIGITS, 512),
    # nn.BatchNorm1d(NUM_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 4))
def fizz_buzz_decode(i, prediction):
    return ["Fizz", "Buzz", "FizzBuzz",str(i)][prediction]


def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])

def main():
    net=newModel().double()
    state_dict=torch.load('./best.pl')
    net.load_state_dict(state_dict['model'])
    # testX = Variable(torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(1, 101)]))
    while True: 
        number = input("Enter a positive integer: ") 
        try: 
            test_num = int(number) 
            if test_num < 0:  # if not a positive int print message and ask for input again 
                print("Sorry, input must be a positive integer, try again") 
                continue 
            break 
        except ValueError: 
            print("That's not an int!")   
            continue   
    # else all is good, val is >=  0 and an integer 

    tmp=np.array(binary_encode(test_num, NUM_DIGITS))
    net.eval()
    testY = net(torch.tensor(tmp).unsqueeze(0).double())
    prediction = testY.max(1)[1]
    print (fizz_buzz_decode(test_num, prediction) )
if __name__ == '__main__':
    main()
