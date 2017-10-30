import numpy as np
import cv2 as cv
import math as m
from random import random as r
from random import gauss as g
from histo import roiMeanSigma

def outOfRange(env, x, y, particle):
	if y < 0 or y >= env[4].shape[0] or x < 0 or x >= env[4].shape[1]:
		if particle is not None:
			env[6].remove(particle)
		return 1
	return 0

def gaussian(x,mu,sigma):
	return 1/(sigma * m.sqrt(2*m.pi))*m.exp(-pow(x-mu,2)/(2*pow(sigma,2)))

class newParticleTracker:
	nbPart = 50
	def __init__(self, x, y, w2, h2, histVar):
		self.x = x
		self.y = y
		self.w2 = w2
		self.h2 = h2
		self.histVar = histVar
		self.particle = self.randParticles()

	def refreshHisto(self, frame):
		roi = frame[int(self.y-6):int(self.y+7), int(self.x-2):int(self.x+3)]
		cv.imshow("refresh", roi)
		histVar = roiMeanSigma(roi)	

	def randParticles(self):
		ret = (np.random.rand(self.nbPart, 3)).astype(np.float32)
		ret[...,0] *= self.w2 * 2 - 1
		ret[...,0] += self.x - self.w2 + 1
		ret[...,1] *= self.h2 * 2 - 1
		ret[...,1] += self.y - self.h2 + 1
		return ret
	
	def moveP(self, prvs, nxt, env):
		a,b,c,d = self.y-self.h2+1,self.y+self.h2,self.x-self.w2+1,self.x+self.w2
		if outOfRange(env, c, a, self) or outOfRange(env, d, b, self):
			return -42
		else:
			flow = cv.calcOpticalFlowFarneback(prvs[a:b,c:d],nxt[a:b,c:d], None, 0.5, 3, 15, 3, 5, 1.2, 0)
			dx = np.mean(flow[...,0])
			stdx = np.std(flow[...,0])
			dy = np.mean(flow[...,1])
			stdy = np.std(flow[...,1])
			for i in range(self.nbPart):
				self.particle[i][0] += g(dx, 3*stdx + 3.0)
				self.particle[i][1] += g(dy, 3*stdy + 3.0)
		return 1

	def gaussianWeight(self, env):
		bgr = env[4]
		i = 0
		for p in self.particle:
			x,y,w = p.ravel()
			x,y = int(x), int(y)
			w = 1.0
			if outOfRange(env, x, y, self):
				return -42
			else:	
				w *= gaussian(bgr[y][x][1], self.histVar[0][0], self.histVar[0][1])
				w *= gaussian(bgr[y][x][1], self.histVar[1][0], self.histVar[1][1])
				w *= gaussian(bgr[y][x][2], self.histVar[2][0], self.histVar[2][1])
				self.particle[i][2] = w
				i += 1
		return 1
	
	def calcParticleDispersion(self, frame):
		roi = frame[self.y-self.h2+1:self.y+self.h2,self.x-self.w2+1:self.x+self.w2]
		hsvRoi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
		hist = cv.calcHist([hsvRoi], [0,1], None, [180,256], [0,180,0,256])
		return hist[120][255]
	
	def resample(self):
		index = int(r() * self.nbPart)
		beta = 0.0
		mw = max(self.particle[...,2])
	
		tmp = (np.random.rand(self.nbPart, 3)).astype(np.float32)
		for i in range(self.nbPart):
			beta += r() * 2.0 * mw
	
			while beta > self.particle[index][2]:
				beta -= self.particle[index][2]
				index = (index + 1) % self.nbPart
			
			tmp[i][0] = self.particle[index][0]
			tmp[i][1] = self.particle[index][1]
		self.particle = tmp

	def draw(self, env):
		pX, pY = 0, 0
		for p in self.particle:
			x,y,w = p.ravel()
			pX += x
			pY += y
			cv.circle(env[4],(int(x),int(y)),0,(255,0,0),-1)
		self.x = int(pX)/self.nbPart
		self.y = int(pY)/self.nbPart
		a,b,c,d = self.y-self.h2,self.y+self.h2,self.x-self.w2,self.x+self.w2
		cv.rectangle(env[4], (c,a), (d,b), (255,0,255), 1)
