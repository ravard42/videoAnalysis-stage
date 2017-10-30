import cv2 as cv
from mouse import focus
from keyboard import loop
import time 
from particleClass import newParticleTracker 

winSize = 600
#cap = cv.VideoCapture(0)
#time.sleep(0.5)
#cap.set(3, winSize)
#cap.set(4, winSize)
cap = cv.VideoCapture("../foot.mp4")


#env = [x,y,w/2,h/2,bgrFrame,roiMeanSigmaTmp,particuleTrackers]
env = [-42,-42,8,22,None,None,[]]
cv.namedWindow('win', 0)
cv.setMouseCallback('win', focus, env)

prvs = None
nxt = None

while 1:
	if nxt is not None:
		prvs = nxt
	ret, env[4] = cap.read()
	if not ret:
		break
	env[4] = cv.resize(env[4], (winSize, winSize))
	nxt = cv.cvtColor(env[4],cv.COLOR_BGR2GRAY)
	if len(env[6]) != 0:
		for pTrack in env[6]:
			if pTrack.moveP(prvs, nxt, env) != -42:
				if pTrack.gaussianWeight(env) != -42:
					pTrack.resample()
		for pTrack in env[6]:
			pTrack.draw(env)
	
	a,b,c,d = env[1]-env[3],env[1]+env[3],env[0]-env[2],env[0]+env[2]
	if env[5] is None:
		cv.rectangle(env[4], (c,a), (d,b), (0,255,0), 1)
	else:
		cv.rectangle(env[4], (c,a), (d,b), (255,0,0), 1)
	cv.imshow('win', env[4])
	k = cv.waitKey(30) & 0xFF
	if not loop(k, env):
		break

#cap.release()
cv.destroyAllWindows()
