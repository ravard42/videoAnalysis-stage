import cv2 as cv
import numpy as np
from focus import foc
from focus import forward
from focus import refreshTrainKPs
from skimage.feature import hog
from skimage import color

cap = cv.VideoCapture("../foot.mp4")

#env = [x,y,w/2,h/2,frame,mask,queryKps,queryDes, trainKps, trainDes, tmpTKps, tmpTDes]
env = [-42,-42,8,22,None,None,None,None, None, None, None, None]
cv.namedWindow('win', 0)
cv.setMouseCallback('win', foc, env)

pas = 8
coef = 4
old = (5,11,17)

def test(x):
	x = 5

i = -1

while 1:
	ret, env[4] = cap.read()
	env[4] = cv.resize(env[4], (640, 480))
	i += 1
	if env[8] is not None:
		if  i%old[0]==0 or i%old[1]==0 or i%old[2]==0:
			refreshTrainKPs(env, i)
		forward(env)
		a = env[1]-env[3]
		b = env[1]+env[3]
		c = env[0]-env[2]
		d = env[0]+env[2]
		cv.rectangle(env[4], (c,a), (d,b), (0,255,0), 1)
	cv.imshow('win', env[4])
	k = cv.waitKey(30) & 0xFF
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
	if k == ord('s'):
		print "<---MATCHING-RESULTS-->"
		print env[10][0]
		print env[10][1]
cv.destroyAllWindows()
