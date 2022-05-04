import torch
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader

class Data(Dataset):
    def __init__(self,X,y):
        # x_tmp=torch.from_numpy(X_train)
        # y_tmp=torch.from_numpy(y_train)
        self.x=torch.tensor(X,dtype=torch.int64)
        self.y=torch.tensor(y,dtype=torch.int64)
        self.len=self.x.shape[0]
    def __getitem__(self,index):      
        return self.x[index], self.y[index]
    def __len__(self):
        return self.len
# data_set=Data()
n_epochs=1000
X_train=pd.read_csv('X_train.csv')
y_train=pd.read_csv('y_train.csv')
X_val=pd.read_csv('X_val.csv')
y_val=pd.read_csv('y_val.csv')
trainloader=DataLoader(dataset=Data(X_train,y_train),batch_size=512)
valloader=DataLoader(dataset=Data(X_val,y_val),batch_size=512)
model = torch.nn.Sequential(
    torch.nn.Linear(NUM_DIGITS, 512),
    # nn.BatchNorm1d(NUM_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 4))
if not os.path.exists('./Model'):
   os.mkdir('./Model')
#n_epochs
best_acc=0
model=model.double().cuda()
optimizer=torch.optim.Adam(model.parameters(), lr=0.0015496760606076839,weight_decay=5e-4)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=10, factor=0.75,verbose=False)
for epoch in range(n_epochs):
    model.train()
    val_loss_ls=[]
    loss_list=[]
    for x, y in tqdm(trainloader):        
        x=x.double().cuda()
        y=y.cuda()
        #clear gradient 
        optimizer.zero_grad()
        #make a prediction 
        z=model(x)
        # calculate loss, da Cross Entropy benutzt wird muss ich in den loss Klassen vorhersagen, 
        # also Wahrscheinlichkeit pro Klasse. Das mach torch.max(y,1)[1])
        loss=criterion(z,y)
        # calculate gradients of parameters 
        loss.backward()
        # update parameters 
        optimizer.step()    
        loss_list.append(loss.item())
    lr_scheduler.step(loss)
    model.eval() 
    y_pred_ls=[]
    y_true_ls=[]
    with torch.no_grad(): 
        for x, y in tqdm(valloader):            
            x=x.double().cuda()
            y=y.cuda()
            z=model(x)
            val_loss=criterion(z,y)
            val_loss_ls.append(val_loss.item())
            y_pred_ls.append(z.argmax(dim=1).cpu())
            y_true_ls.append(y.cpu())
    y_pred=np.concatenate(y_pred_ls)
    y_true=np.concatenate(y_true_ls)
    val_acc=accuracy_score(y_true, y_pred)

    y_pred_tr_ls=[]
    y_true_tr_ls=[]
    with torch.no_grad(): 
        for x, y in tqdm(trainloader):            
            x=x.double().cuda()
            y=y.cuda()
            z=model(x)
            y_pred_tr_ls.append(z.argmax(dim=1).cpu())
            y_true_tr_ls.append(y.cpu())
    y_pred_tr=np.concatenate(y_pred_tr_ls)
    y_true_tr=np.concatenate(y_true_tr_ls)
    train_acc=accuracy_score(y_true_tr, y_pred_tr)

    print('epoch {}, loss {}'.format(epoch, np.mean(loss_list)))
    print('acc',val_acc)


    if val_acc>best_acc:
          best_acc=val_acc
          torch.save({"best_acc":best_acc,"model":model.state_dict()},'./Model/best.pl')
          print(f"best_acc: {best_acc}")
    # print(acc)
