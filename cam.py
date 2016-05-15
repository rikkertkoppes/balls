'''
This is how to track a white ball example using SimpleCV
The parameters may need to be adjusted to match the RGB color
of your object.
The demo video can be found at:
http://www.youtube.com/watch?v=jihxqg3kr-g
'''
print __doc__

import SimpleCV
import cv2
import cv2.cv as cv
import numpy as np

img = cv2.imread('20160514_165716.jpg')
img = cv2.resize(img,(800,450))
bimg = cv2.medianBlur(img,51)
cimg = cv2.cvtColor(bimg,cv2.COLOR_BGR2GRAY)
# th, cimg = cv2.threshold(cimg,210,255,cv2.THRESH_BINARY)
edges = cv2.Canny(cimg,200,200)


circles = cv2.HoughCircles(cimg,cv.CV_HOUGH_GRADIENT,1,20,
	param1=200,param2=40,minRadius=0,maxRadius=0)

print circles

if circles is not None:
	circles = np.uint16(np.around(circles))
	for i in circles[0,:]:
	    # draw the outer circle
	    cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
	    # draw the center of the circle
	    cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# display = SimpleCV.Display()
# cam = SimpleCV.Camera()
# normaldisplay = True

# ct = 160

# img = SimpleCV.Image('20160514_165716.jpg').rotate(-90).resize(640);

# while display.isNotDone():

# 	if display.mouseRight:
# 		normaldisplay = not(normaldisplay)
# 		print "Display Mode:", "Normal" if normaldisplay else "Segmented" 
	
# 	# img = cam.getImage().flipHorizontal()
# 	dist = img.colorDistance(SimpleCV.Color.BLACK).dilate(2)
# 	segmented = dist.stretch(200,255)
# 	# edges = img.edges(t1=160)
# 	circles = img.findCircle(canny=ct,thresh=200,distance=15)
# 	if circles is not None:
# 		# circles = circles.sortArea()
# 		circles.draw(width=4)
# 		if circles.length > 1:
# 			# raise threshold
# 			ct = min(300,ct+1)
# 		# else:
# 			# circles.draw(width=4)
# 	else:
# 		# lower threshold
# 		ct = max(50,ct-1)
# 	img.drawText("{} {}".format(ct,circles),10,10)
# 	# # blobs = segmented.findBlobs()
# 	# # if blobs:
# 	# # 	circles = blobs.filter([b.isCircle(0.2) for b in blobs])
# 	# # 	if circles:
# 	# # 		img.drawCircle((circles[-1].x, circles[-1].y), circles[-1].radius(),SimpleCV.Color.BLUE,3)

# 	if normaldisplay:
# 		img.show()
# 	else:
# 		segmented.show()