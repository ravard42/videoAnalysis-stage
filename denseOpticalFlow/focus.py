import cv2 as cv
import numpy as np

def foc(event, x, y, flags, env):
	if event == cv.EVENT_FLAG_LBUTTON:
		env[0], env[1] = x, y
		env[5] = np.zeros_like(env[4])
		env[5][...,1] = 255
