'''
This is how to track a white ball example using SimpleCV
The parameters may need to be adjusted to match the RGB color
of your object.
The demo video can be found at:
http://www.youtube.com/watch?v=jihxqg3kr-g
'''
print __doc__

import SimpleCV

display = SimpleCV.Display()
cam = SimpleCV.Camera()
normaldisplay = True

ct = 160

img = SimpleCV.Image('20160514_165716.jpg').rotate(-90).resize(640);

while display.isNotDone():

	if display.mouseRight:
		normaldisplay = not(normaldisplay)
		print "Display Mode:", "Normal" if normaldisplay else "Segmented" 
	
	# img = cam.getImage().flipHorizontal()
	dist = img.colorDistance(SimpleCV.Color.BLACK).dilate(2)
	segmented = dist.stretch(200,255)
	# edges = img.edges(t1=160)
	circles = img.findCircle(canny=ct,thresh=200,distance=15)
	if circles is not None:
		# circles = circles.sortArea()
		circles.draw(width=4)
		if circles.length > 1:
			# raise threshold
			ct = min(300,ct+1)
		# else:
			# circles.draw(width=4)
	else:
		# lower threshold
		ct = max(50,ct-1)
	img.drawText("{} {}".format(ct,circles),10,10)
	# # blobs = segmented.findBlobs()
	# # if blobs:
	# # 	circles = blobs.filter([b.isCircle(0.2) for b in blobs])
	# # 	if circles:
	# # 		img.drawCircle((circles[-1].x, circles[-1].y), circles[-1].radius(),SimpleCV.Color.BLUE,3)

	if normaldisplay:
		img.show()
	else:
		segmented.show()