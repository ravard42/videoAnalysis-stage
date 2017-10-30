import numpy as np
import cv2 as cv
import time

cap = cv.VideoCapture('../foot.mp4')
fgbg = cv.bgsegm.createBackgroundSubtractorMOG(history=3,backgroundRatio=0.1, nmixtures=3, noiseSigma = 7)
minArea = 5

while(1):
	ret, frame = cap.read()
	fgmask = fgbg.apply(frame)
	fgmask = cv.GaussianBlur(fgmask, (15, 15), 0)
	fgmask = cv.threshold(fgmask, 42, 255, cv.THRESH_BINARY)[1]
	#fgmask = cv.erode(fgmask,None,iterations = 1)
	fgmask = cv.dilate(fgmask, None, iterations= 2)
	img, contours, hierarchy = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	for c in contours:
		if cv.contourArea(c) < minArea:
			continue
		(x, y, w, h) = cv.boundingRect(c)
		cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
	cv.imshow('frame',frame)
	cv.imshow('fgbg',fgmask)
	k = cv.waitKey(1) & 0xff
	if k == ord('q'):
		break
	if k == ord('p'):
		while cv.waitKey(1) != ord('p'):
			continue
		
cap.release()
cv.destroyAllWindows()
