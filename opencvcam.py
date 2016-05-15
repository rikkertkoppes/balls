import cv2
import math
import cv2.cv as cv
import numpy as np
import numpy.ma as ma

from sklearn.externals import joblib 
from skimage.feature import hog 

def detect(img, threshold):
    bimg = cv2.medianBlur(img,11)
    cimg = cv2.equalizeHist(bimg)
    h,w, = img.shape
    hw = h / 2
    hh = w / 2
    # cimg = cv2.cvtColor(bimg,cv2.COLOR_BGR2GRAY)
    # roi = bimg
    roi = None
    circles = cv2.HoughCircles(cimg,cv.CV_HOUGH_GRADIENT,1,20,
        param1=threshold,param2=40,minRadius=0,maxRadius=0)

    if circles is not None:
        circles = np.uint16(np.around(circles))[0,:]
        circles = sorted(circles,key=lambda c: (c[0]-hw)**2 + (c[1]-hh)**2)

        # pick the one closest to center
        c = circles[0]
        x = c[0]-c[2]
        y = c[1]-c[2]
        d = 2*c[2]
        roi = img.copy()[y:y+d,x:x+d]
        cv2.circle(img,(c[0],c[1]),c[2],(0,255,0),2)
        cv2.line(img,(c[0]-c[2],c[1]),(c[0]+c[2],c[1]),(0,255,0),1)

        #resize to create an image we can segment later
        roi = cv2.resize(roi,(255,255))

    return roi, circles

def identify(img, clf):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.resize(img,(28,28))
    roi_hog_fd = hog(img, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
    return nbr[0]


def segment(img):
    #threshold
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 127, 40)
    
    #mask out outside the ball
    mask = img.copy()
    mask.fill(255)
    cv2.circle(mask, (127,127), 120, 0, -1)
    img = cv2.bitwise_or(img,mask)

    inv = (255 - img)
    # inv = img.copy()

    contours, hierarchy = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    draw = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    imgs = []
    # bboxes sorted by x
    rects = sorted([cv2.boundingRect(ctr) for ctr in contours],key=lambda ctr: ctr[0])
    # print len(rects)
    # print rects[:5]
    for (x,y,w,h) in rects:
        #resize rois to 28x28 for further recognition against mnist
        if y > (128-h) and y < 128:
            imgs.append(cv2.resize(img[y:y+h,x:x+w], (28,28)))
            cv2.rectangle(draw,(x,y),(x+w,y+h),(0,255,0),2)
        
    return draw, imgs[:5]

def static():
    img = cv2.imread('20160514_165716.jpg')
    img = cv2.resize(img,(800,450))

    roi, circles = detect(img, 200)
    print circles

    cv2.imshow('detected circles',img)
    cv2.imshow('detected roi',roi)
    cv2.waitKey(0)



def run():
    cap = cv2.VideoCapture(0)
    clf = joblib.load("digits_cls.pkl")
    th = 200
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        frame = cv2.flip(frame,1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame = cv2.equalizeHist(frame)

        #detect, adaptive
        roi, circles = detect(frame, 200)
        if circles is None:
            th = max(0,th-1)
        else:
            # print circles
            if len(circles) > 1:
                th = min(1000,th+1)

        # Display the resulting frame
        cv2.putText(frame,'{} {}'.format(th,circles),(10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
        cv2.imshow('frame',frame)

        if roi is not None:
            #flip back
            roi = cv2.flip(roi,1)
            #segment
            roi, rois = segment(roi) 

            nr = ''.join([str(identify(r, clf)) for r in rois])
            cv2.putText(roi,'{}'.format(nr),(10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0)
            
            cv2.imshow('roi',roi)
            cv2.moveWindow('roi',700,0)

            # print len(rois)
            if len(rois) == 5:
                num = np.concatenate([r for r in rois], axis=1)
                cv2.imshow('rois',num)
                cv2.moveWindow('rois',700,300)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()

run()
# static()
cv2.destroyAllWindows()
