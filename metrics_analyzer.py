import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;

rootpath = 'D:/python/workplaceDetector';

df = pd.read_csv(rootpath + '/metrics.csv');
df = df.iloc[:,2:];
df.index = pd.RangeIndex(start=1,stop=41)

#[tp,fp,fn,tn]
df['cm'] = df['cm'].apply(lambda s: [int(x) for x in s.replace('[','').replace(']','').split(',')]);
df['val_cm'] = df['val_cm'].apply(lambda s: [int(x) for x in s.replace('[','').replace(']','').split(',')]);

#CrossEntropyLoss
plt.plot(df['val_cel'][1:], label = "test");
plt.plot(df['cel'][1:], label = "train");
plt.title('CrossEntropy Loss');
plt.xlabel('epoch');
plt.grid(visible=True);
plt.legend();
plt.show();

#Accuracy
#(tp+tn)/(tp+tn+fp+fn)
df['acc'] = df['cm'].apply(lambda x: (x[0] + x[3])/(np.sum(x)));
df['val_acc'] = df['val_cm'].apply(lambda x: (x[0] + x[3])/(np.sum(x)));

#Precision
#tp/(tp+fp)
df['prec'] = df['cm'].apply(lambda x: x[0]/(x[0]+x[1]));
df['val_prec'] = df['val_cm'].apply(lambda x: x[0]/(x[0]+x[1]));

#Recall
#tp/(tp+fn)
df['rec'] = df['cm'].apply(lambda x: x[0]/(x[0]+x[2]));
df['val_rec'] = df['val_cm'].apply(lambda x: x[0]/(x[0]+x[2]));

#F1
#2*prec*rec/(prec+rec)
df['f1'] = 2*df['prec']*df['rec']/(df['prec']+df['rec']);
df['val_f1'] = 2*df['val_prec']*df['val_rec']/(df['val_prec']+df['val_rec']);

df.to_csv(rootpath + '/metrics_calculated.csv');

figure, axis = plt.subplots(2, 2);

axis[0,0].plot(df['val_acc'][1:], label = "test");
axis[0,0].plot(df['acc'][1:], label = "train");
axis[0,0].set_title('Accuracy');
axis[0,0].set_xlabel('epoch');
axis[0,0].grid(visible=True);
axis[0,0].legend();

axis[0,1].plot(df['val_prec'][1:], label = "test");
axis[0,1].plot(df['prec'][1:], label = "train");
axis[0,1].set_title('Precision');
axis[0,1].set_xlabel('epoch');
axis[0,1].grid(visible=True);
axis[0,1].legend();

axis[1,1].plot(df['val_rec'][1:], label = "test");
axis[1,1].plot(df['rec'][1:], label = "train");
axis[1,1].set_title('Recall');
axis[1,1].set_xlabel('epoch');
axis[1,1].grid(visible=True);
axis[1,1].legend();

axis[1,0].plot(df['val_f1'][1:], label = "test");
axis[1,0].plot(df['f1'][1:], label = "train");
axis[1,0].set_title('F1');
axis[1,0].set_xlabel('epoch');
axis[1,0].grid(visible=True);
axis[1,0].legend();

plt.show();




