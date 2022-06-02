import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;

rootpath = 'D:/python/workplaceDetector';

ROC = pd.read_csv(rootpath + '/ROC.csv');
ROC.index = ROC.iloc[:,0];
ROC = ROC.iloc[:,1:];



plt.plot(ROC['fpr17'],ROC['tpr17'], label = "17");
plt.plot(ROC['fpr19'],ROC['tpr19'], label = "19");
plt.title('ROC');
plt.xlabel('False Positive Rate');
plt.ylabel('True Positive Rate');
plt.grid(visible=True);
plt.legend();
plt.show();





plt.plot(ROC['f117'], label = "17");
plt.plot(ROC['f119'], label = "19");
plt.title('F1');
plt.xlabel('Threshold');
plt.grid(visible=True);
plt.legend();
plt.show();



