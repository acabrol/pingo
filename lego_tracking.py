# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
colorLower = (110,50,50)
colorUpper = (130,255,255)
pts = deque(maxlen=args["buffer"])

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	camera = cv2.VideoCapture(0)

# otherwise, grab a reference to the video file
else:
	camera = cv2.VideoCapture(args["video"])

# keep looping
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()

	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if args.get("video") and not grabbed:
		break

	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=600)
	# blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, colorLower, colorUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
	center = None

	# only proceed if at least one contour was found
	if len(cnts) > 0:
		#cnt = max(cnts, key=cv2.contourArea)
		boxes=[]
		for cnt in cnts:
			if cv2.contourArea(cnt)>500:
				print(cv2.contourArea(cnt))
				approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
				bx,by,bw,bh = cv2.boundingRect(cnt)
				cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),(255,0,0),3) # draw rectangle in blue color)
				edges = cv2.Canny(cnt,100,200)
				for edge in edges:
					cv2.drawContours(frame,[edge],0,(0,255,0),-1
				# if len(approx)==5:
				#     print "pentagon"
				#     cv2.drawContours(frame,[cnt],0,255,-1)
				# elif len(approx)==3:
				#     print "triangle"
				#     cv2.drawContours(frame,[cnt],0,(0,255,0),-1)
				# elif len(approx)==4:
				#     print "square"
				#     cv2.drawContours(frame,[cnt],0,(0,0,255),-1)
				# elif len(approx) == 9:
				#     print "half-circle"
				#     cv2.drawContours(frame,[cnt],0,(255,255,0),-1)
				# elif len(approx) > 15:
				#     print "circle"
				#     cv2.drawContours(frame,[cnt],0,(0,255,255),-1)
		#cv2.drawContours(frame,boxes,0,(0,0,255),2)

	# show the frame to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
