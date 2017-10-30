import cv2 as cv
import numpy as np
from focus import foc
import time
import math as m


#cap = cv.VideoCapture(0)
#time.sleep(0.5)
#cap.set(3, 640)
#cap.set(4, 480)
cap = cv.VideoCapture("../foot.mp4")

#env = [x,y,w/2,h/2,frame,hsv]
env = [-42,-42,32,32,None,None]
cv.namedWindow('win', 0)
cv.setMouseCallback('win', foc, env)

ret, first = cap.read()
first = cv.resize(first, (640, 480))
prvs = cv.cvtColor(first,cv.COLOR_BGR2GRAY)
env[5] = np.zeros_like(first)
env[5][...,1] = 255

pas = 8
coef = 4

while(1):
	ret, env[4] = cap.read()
	env[4] = cv.resize(env[4], (640, 480))
	if env[0] != -42:
		foc(cv.EVENT_FLAG_LBUTTON, env[0], env[1], None, env)
		next = cv.cvtColor(env[4],cv.COLOR_BGR2GRAY)
		a = env[1]-env[3]
		b = env[1]+env[3]
		c = env[0]-env[2]
		d = env[0]+env[2]
		flow = cv.calcOpticalFlowFarneback(prvs[a:b,c:d],next[a:b,c:d], None, 0.5, 3, 15, 3, 5, 1.2, 0)
		mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
		pol = np.zeros(flow.shape)
		pol[...,0] = mag
		pol[...,1] = ang
		
		env[5][a:b,c:d,2] = cv.normalize(pol[...,0],None,0,255,cv.NORM_MINMAX)
		env[5][a:b,c:d,0] = pol[...,1]*180/(m.pi*2)
		rgb = cv.cvtColor(env[5],cv.COLOR_HSV2BGR)
		cv.imshow('opticalFlow',rgb)
		prvs = next
		
		maskPol = pol[pol[...,0] > 0.8]
		if maskPol.any():
			magM = np.mean(maskPol[...,0])
			angM = (np.mean(np.cos(maskPol[...,1])), np.mean(np.sin(maskPol[...,1])))
			env[0] += angM[0] * magM * 1.2
			env[0] = int(env[0])
			env[1] += angM[1] * magM * 1.2
			env[1] = int(env[1])
		cv.rectangle(env[4], (c,a), (d,b), (0,255,0), 1)
	cv.imshow('win', env[4])
	
	k = cv.waitKey(30) & 0xff
	
	#if k != 255:
	#	print k
	if k == ord('q'):
	    break
	if k == 57 and (pas == 8 or pas == coef * 8):
			pas = coef * 8 if pas == 8 else 8
	if k == 55 and (pas == 8 or pas == 8 / coef):
			pas = 8 / coef if pas == 8 else 8
	if k == 56:
		env[3] += pas
	if k == 50 and env[3] > pas:
		env[3] -= pas
	if k == 52 and env[2] > pas:
		env[2] -= pas
	if k == 54 and env[2]:
		env[2] += pas
	if k == ord('r'):
		foc(cv.EVENT_FLAG_LBUTTON, env[0], env[1], None, env)
	if k == ord('p'):
		print "{},{}".format(env[2], env[3])
	

cap.release()
cv.destroyAllWindows()
