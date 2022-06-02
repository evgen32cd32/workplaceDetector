import os;
import numpy as np;
import pandas as pd;

import torch;
import torch.nn as nn;
import torch.nn.functional as F;
from torchvision.io import read_image;

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
    
    valset = MyDataset(root=valroot, trueFolder = trueFolder, falseFolder = falseFolder);
    valloader = torch.utils.data.DataLoader(valset, batch_size=512,shuffle=False);
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu');
    
    net17 = Net();
    net17.load_state_dict(torch.load(rootpath + '/savestates/17'));
    net19 = Net();
    net19.load_state_dict(torch.load(rootpath + '/savestates/19'));
    
    net17.to(device);
    net19.to(device);
    
    res = pd.DataFrame(columns = ['score','prob17','prob19']);
    
    outputs17 = torch.empty((0,2)).to(device);
    outputs19 = torch.empty((0,2)).to(device);
    score = torch.empty((0)).to(device);
    with torch.no_grad():
        for i,data in enumerate(valloader, 0):
            inputs, scores = data[0].to(device), data[1].to(device);
            outputs17 = torch.cat([outputs17,net17(inputs)]);
            outputs19 = torch.cat([outputs19,net19(inputs)]);
            score = torch.cat([score,scores]);
        o17 = F.softmax(outputs17,dim=1).cpu().numpy();
        o19 = F.softmax(outputs19,dim=1).cpu().numpy();
        res['score'] = score.cpu().numpy()
        res['prob17'] = o17[:,1];
        res['prob19'] = o19[:,1];
    res.to_csv(rootpath + '/17_19_validation.csv');
    ROC = pd.DataFrame(columns=['tp17','fp17','fn17','tn17','tp19','fp19','fn19','tn19']);
    ROC.index.name = 'threshold';
    for t in np.linspace(0.,1.,101):
        ROC.loc[t,'tp17'] = ((res['score'] == 1) & (res['prob17'] >= t)).sum();
        ROC.loc[t,'fp17'] = ((res['score'] == 0) & (res['prob17'] >= t)).sum();
        ROC.loc[t,'fn17'] = ((res['score'] == 1) & (res['prob17'] < t)).sum();
        ROC.loc[t,'tn17'] = ((res['score'] == 0) & (res['prob17'] < t)).sum();
        ROC.loc[t,'tp19'] = ((res['score'] == 1) & (res['prob19'] >= t)).sum();
        ROC.loc[t,'fp19'] = ((res['score'] == 0) & (res['prob19'] >= t)).sum();
        ROC.loc[t,'fn19'] = ((res['score'] == 1) & (res['prob19'] < t)).sum();
        ROC.loc[t,'tn19'] = ((res['score'] == 0) & (res['prob19'] < t)).sum();
    
    ROC['prec17'] = ROC['tp17']/(ROC['tp17']+ROC['fp17']);
    ROC['rec17'] = ROC['tp17']/(ROC['tp17']+ROC['fn17']);
    ROC['prec19'] = ROC['tp19']/(ROC['tp19']+ROC['fp19']);
    ROC['rec19'] = ROC['tp19']/(ROC['tp19']+ROC['fn19']);
    ROC['f117'] = 2*ROC['prec17']*ROC['rec17']/(ROC['prec17']+ROC['rec17']);
    ROC['f119'] = 2*ROC['prec19']*ROC['rec19']/(ROC['prec19']+ROC['rec19']);
    
    #All the NaNs should be interpreted as zeroes
    ROC = ROC.fillna(0);
    
    ROC['tpr17'] = ROC['tp17']/(ROC['tp17']+ROC['fn17']);
    ROC['fpr17'] = ROC['fp17']/(ROC['fp17']+ROC['tn17']);
    ROC['tpr19'] = ROC['tp19']/(ROC['tp19']+ROC['fn19']);
    ROC['fpr19'] = ROC['fp19']/(ROC['fp19']+ROC['tn19']);
    
    ROC.to_csv(rootpath + '/ROC.csv');


if __name__ == "__main__":
    main();


