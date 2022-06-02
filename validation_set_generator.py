import os;
import numpy as np;

rootpath = 'D:/python/workplaceDetector';

root = rootpath + '/workplace/';
trueFolder = 'true/';
falseFolder = 'false/';

validationFolder = 'validation/';

ti = os.listdir(root + trueFolder);
fi = os.listdir(root + falseFolder);

valti = np.random.choice(ti,int(np.floor(len(ti)*0.1)),replace=False);
valfi = np.random.choice(fi,int(np.floor(len(fi)*0.1)),replace=False);


for imgname in valti:
    os.replace(root + trueFolder + imgname, root + validationFolder + trueFolder + imgname);

for imgname in valfi:
    os.replace(root + falseFolder + imgname, root + validationFolder + falseFolder + imgname);


