import cv2
import math
import cv2.cv as cv
import numpy as np
import numpy.ma as ma
import deepMnist
import time
import urllib2

from scipy import ndimage

def detect(img, threshold):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # paint detection rectangle, only detect inside that
    cv2.rectangle(img,(245,165),(395,315),(0,255,0),2)
    gray = gray[165:315,245:395]

    bimg = cv2.medianBlur(gray,11)
    cimg = cv2.equalizeHist(bimg)
    h,w,c = img.shape
    hw = h / 2
    hh = w / 2

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
        patch = gray.copy()[y:y+d,x:x+d]
        cv2.circle(img,(245+c[0],165+c[1]),c[2],(0,255,0),2)
        cv2.line(img,(245+c[0]-c[2],165+c[1]),(245+c[0]+c[2],165+c[1]),(0,255,0),1)

        #resize to create an image we can segment later
        if patch.size > 0:
            roi = cv2.resize(patch,(255,255))

    return img, roi, circles

# see http://openmachin.es/blog/tensorflow-mnist
# and http://openmachin.es/blog/tensorflow-mnist-nod
# resize retaining proportions, to a 20x20 box and 4 px border
#resize rois to 28x28 for further recognition against mnist
def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

def segment(img):
    #threshold
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 127, 45)
    # ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

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
        if y > (128-h) and y < 128:
            # 28 x 28 patch
            patch = np.zeros((28, 28), np.uint8)
            patch.fill(255)
            # digit from original, scale to max 20px largest dimension
            digit = img[y:y+h,x:x+w]
            maxdim = max(h,w)
            scale = 20.0 / maxdim
            dw = int(round(w*scale))
            dh = int(round(h*scale))
            digit = cv2.resize(digit, (dw,dh))
            # paste digit in patch
            ox = (28 - dw)/2
            oy = (28 - dh)/2
            patch[oy:oy+dh, ox:ox+dw] = digit
            patch = 255 - patch

            #shift center of mass
            shiftx,shifty = getBestShift(patch)
            shifted = shift(patch,shiftx,shifty)
            patch = shifted

            imgs.append(patch)
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

def beep():
    print "\a"

def showCase(nr):
    url = 'http://localhost:8477/show/{}'.format(nr)
    urllib2.urlopen(url)
    beep()

def run():
    cap = cv2.VideoCapture(0)
    deepMnist.restore()
    th = 200
    lastScans = []
    lastResult = None
    lastScanTime = None
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.flip(frame,1)

        #detect, adaptive
        frame, roi, circles = detect(frame, 200)

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

            # print len(rois)
            # we have 5 digits
            if len(rois) == 5:
                # predict number
                flat = [r.flatten() for r in rois]
                result = deepMnist.predict(flat)
                nr = int(''.join([str(r) for r in result]))
                lastScans.append(nr)
                # keep last 5
                lastScans = lastScans[-5:]
                print lastScans

                #scanned the same number 5 times in a row
                if len(lastScans) == 5:
                    if lastScans[1:] == lastScans[:-1]:
                        if not lastResult == lastScans[0]:
                            print 'scanResult'
                            print lastScans[0]
                            lastResult = lastScans[0]
                            lastScanTime = time.clock()
                            showCase(lastResult)


                cv2.putText(roi,'{}'.format(nr),(10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0)

                num = np.concatenate([r for r in rois], axis=1)
                cv2.imshow('rois',num)
                cv2.moveWindow('rois',700,300)

            cv2.imshow('roi',roi)
            cv2.moveWindow('roi',700,0)

        # reset scanner after 5 seconds
        if lastScanTime is not None:
            if time.clock() - lastScanTime > 5:
                print 'scan reset'
                lastScanTime = None
                lastResult = None

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    deepMnist.close()

run()
# static()
cv2.destroyAllWindows()
