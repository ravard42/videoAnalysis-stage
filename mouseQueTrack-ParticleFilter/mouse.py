import cv2 as cv
import numpy as np
from particleClass import outOfRange


from histo import roiMeanSigma

from particleClass import newParticleTracker

def focus(event, x, y, flags,  env):
	if event == cv.EVENT_MOUSEMOVE:
		env[0], env[1] = x, y
	if event == cv.EVENT_FLAG_LBUTTON and env[5] is None:
		a,b,c,d = env[1]-env[3]+1,env[1]+env[3],env[0]-env[2]+1,env[0]+env[2]
		if outOfRange(env, c, a, None) or outOfRange(env, d, b, None):
			print "outOfRange"
		else:
			roi = env[4][env[1]-env[3]+1:env[1]+env[3], env[0]-env[2]+1:env[0]+env[2]]
			cv.imshow("histSelected", roi)
			env[5] = roiMeanSigma(roi)
	if event == cv.EVENT_FLAG_RBUTTON and env[5] is not None:
		env[6].append(newParticleTracker(env[0], env[1], env[2], env[3], env[5]))
