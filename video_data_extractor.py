import cv2 as cv;
import numpy as np;

xmin = 115;
xmax = 350;
ymin = 210;
ymax = 445;

xgoal = 119;
ygoal = 119;

vd = cv.VideoCapture('D:/python/data/Camera 3_20220526_003249.mp4');
imgpath = 'D:/python/data/workplace/';
frame_cnt = int(vd.get(cv.CAP_PROP_FRAME_COUNT));
trailing_zeros = int(np.ceil(np.log(frame_cnt)/np.log(10)));

z = '{:>0' + str(trailing_zeros) + '}';



i = 0;
k = 0;

cp = 1;
while vd.isOpened():
    k += 1;
    ret, frame = vd.read();
    if not ret:
        print('done');
        break;
    if k == 15:
        k = 0;
        img = cv.resize(frame[ymin:ymax,xmin:xmax],(xgoal,ygoal));
        fl = cv.imwrite(imgpath + z.format(str(i)) + '.png', img);
        if not fl:
            print('smth is going wrong');
            break;
        i += 1;
        if (i*1500/frame_cnt > cp):
            print(str(cp) + '%');
            cp += 1;
    #cv.imshow("Display window", img);
    #k = cv.waitKey(0);
    #break;


vd.release();
cv.destroyAllWindows();


