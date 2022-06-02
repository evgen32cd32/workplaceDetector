import os;
import numpy as np;
import pandas as pd;

import torch;
import torch.nn as nn;
import torch.nn.functional as F;
from torchvision.io import read_image;

from datetime import datetime;


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, trueFolder, falseFolder, transform=None, target_transform=None):
        self.root = root;
        self.ds = pd.DataFrame([(trueFolder + imgname, 1) for imgname in os.listdir(root + trueFolder)]);
        self.ds = pd.concat([self.ds,pd.DataFrame([(falseFolder + imgname, 0) for imgname in os.listdir(root + falseFolder)])],ignore_index = True);
        self.transform = transform;
        self.target_transform = target_transform;
    
    def __getitem__(self, idx):
        img_path = self.root + self.ds.iloc[idx,0];
        image = read_image(img_path)/255;
        score = self.ds.iloc[idx,1];
        if self.transform:
            image = self.transform(image);
        if self.target_transform:
            score = target_transform(score);
        return image, score;
    
    def __len__(self):
        return len(self.ds);

class Net(nn.Module):
    def __init__(self):
        super().__init__();
        self.conv1 = nn.Conv2d(3,20,20);
        self.pool = nn.MaxPool2d(4);
        self.fc1 = nn.Linear(20*25*25,2);
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)));
        x = torch.flatten(x,1);
        x = F.relu(self.fc1(x));
        return x;


def main():
    
    rootpath = 'D:/python/workplaceDetector';
    
    root = 'D:/python/data/workplace/';
    valroot = root + 'validation/';
    trueFolder = 'true/';
    falseFolder = 'false/';
    
    trainset = MyDataset(root=root, trueFolder = trueFolder, falseFolder = falseFolder);
    valset = MyDataset(root=valroot, trueFolder = trueFolder, falseFolder = falseFolder);
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=512,shuffle=True);
    valloader = torch.utils.data.DataLoader(valset, batch_size=512,shuffle=False);
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu');
    
    net = Net();
    net.load_state_dict(torch.load(rootpath + '/savestates/20'))
    net.to(device);
    criterion = nn.CrossEntropyLoss(reduction='sum');
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9);
    
    res = pd.DataFrame(columns = ['epoch', 'cel', 'cm', 'val_cel','val_cm']);
    
    for epoch in range(20,40):
        now = datetime.now();
        current_time = now.strftime("%H:%M:%S");
        print(current_time + f' Starting {epoch + 1} epoch');
        running_loss = 0;
        #[[tp,fp],[fn,tn]]
        cm = [[0,0],[0,0]];
        for i,data in enumerate(trainloader, 0):
            inputs, scores = data[0].to(device), data[1].to(device);
            optimizer.zero_grad();
            outputs = net(inputs);
            outp = (outputs[:,0]<=outputs[:,1]);
            cm[0][0] += (outp*scores).sum().item();
            cm[1][1] += torch.logical_not(outp+scores).sum().item();
            cm[0][1] += (outp*torch.logical_not(scores)).sum().item();
            cm[1][0] += (torch.logical_not(outp)*scores).sum().item();
            loss = criterion(outputs, scores);
            loss.backward();
            optimizer.step();
            running_loss += loss.item();
            now = datetime.now();
            current_time = now.strftime("%H:%M:%S");
            print(current_time + f' [{epoch + 1}, {(i + 1)*100/len(trainloader):.0f}%] loss: {running_loss:.3f}');
        now = datetime.now();
        current_time = now.strftime("%H:%M:%S");
        print(current_time + f' Epoch {epoch + 1} done. Loss: {running_loss/len(trainloader):.3f}');
        torch.save(net.state_dict(), rootpath + '/savestates/'+str(epoch+1));
        res.loc[epoch,'epoch'] = epoch;
        res.loc[epoch,'cel'] = running_loss/len(trainloader);
        res.loc[epoch,'cm'] = cm;
        print(current_time + ' ' + str(running_loss/len(trainloader)) + ' ' + str(cm));
        with torch.no_grad():
            now = datetime.now();
            current_time = now.strftime("%H:%M:%S");
            print(current_time + f' Starting validation');
            val_loss = 0;
            val_cm = [[0,0],[0,0]];
            for i,data in enumerate(valloader, 0):
                inputs, scores = data[0].to(device), data[1].to(device);
                outputs = net(inputs);
                outp = (outputs[:,0]<=outputs[:,1]);
                val_cm[0][0] += (outp*scores).sum().item();
                val_cm[1][1] += torch.logical_not(outp+scores).sum().item();
                val_cm[0][1] += (outp*torch.logical_not(scores)).sum().item();
                val_cm[1][0] += (torch.logical_not(outp)*scores).sum().item();
                val_loss += criterion(outputs, scores).item();
                now = datetime.now();
                current_time = now.strftime("%H:%M:%S");
                print(current_time + f' [{epoch + 1}, {(i + 1)*100/len(valloader):.0f}%] loss: {val_loss:.3f}');
            res.loc[epoch,'val_cel'] = val_loss/len(valloader);
            res.loc[epoch,'val_cm'] = val_cm;
            print(current_time + ' ' + str(val_loss/len(valloader)) + ' ' + str(val_cm));
    res.to_csv(rootpath + '/metrics2.csv');

if __name__ == "__main__":
    main();



