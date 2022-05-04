import torch.nn as nn
import numpy as np
import torch
import sys
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
NUM_DIGITS=20
class newModel(nn.Module):
    def __init__(self, vocab_size=26):
        super().__init__()
        self.hidden_dim = 256
        self.batch_size = 256
        self.emb_dim = NUM_DIGITS
        
        # self.embedding = nn.Embedding(vocab_size, self.emb_dim, padding_idx=0)
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=2)
        # self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        # self.gmlp_t=gMLP(num_tokens = 1000,dim = 32, depth = 2,  seq_len = 40, act = nn.Tanh())
        self.gru = nn.GRU(self.emb_dim, self.hidden_dim, num_layers=6, 
                               bidirectional=True, dropout=0.20)
        
        
        self.block1=nn.Sequential(nn.Linear(3584,1024),
                                            nn.BatchNorm1d(1024),
                                            nn.LeakyReLU(),
                                            nn.Linear(1024,512),
                                            nn.BatchNorm1d(512),
                                            nn.LeakyReLU(),
                                            nn.Linear(512,256),
                                 )

        self.block2=nn.Sequential(
                                               nn.BatchNorm1d(256),
                                               nn.LeakyReLU(),
                                               nn.Linear(256,128),
                                               nn.BatchNorm1d(128),
                                               nn.LeakyReLU(),
                                               nn.Linear(128,64),
                                               nn.BatchNorm1d(64),
                                               nn.LeakyReLU(),
                                               nn.Linear(64,4)
                                            )
        
    def forward(self, x):
        # x=self.embedding(x)
        # output=self.transformer_encoder(x).permute(1, 0, 2)
        # output=self.gmlp_t(x).permute(1, 0, 2)
        x=x.view(1,x.shape[0],x.shape[1])
        # output=self.gmlp_t(x).permute(1, 0, 2)
        # print(output.shape)
        output,hn=self.gru(x)
        output=output.permute(1,0,2)
        hn=hn.permute(1,0,2)
        output=output.reshape(output.shape[0],-1)
        hn=hn.reshape(output.shape[0],-1)
        output=torch.cat([output,hn],1)
        # print('output.shape',output.shape)
        output=self.block1(output)
        return self.block2(output)

def fizz_buzz_decode(i, prediction):
    return ["Fizz", "Buzz", "FizzBuzz",str(i)][prediction]


def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])

def main():
    net=newModel().double()
    state_dict=torch.load('/content/Model/best.pl')
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