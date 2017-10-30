import cv2 as cv
import math as m

def histMean(hist):
	nbPix = int(sum(hist)[0])
	mean = 0
	for i in range(len(hist)):
		mean += i * hist[i][0]
	mean /= nbPix
	return mean

def histDeviation(hist, mean):
	nbPix = int(sum(hist)[0])
	dev = 0
	for i in range(len(hist)):
		dev += pow(i - mean, 2) * hist[i][0]
	dev /= nbPix
	return m.sqrt(dev)

def roiMeanSigma(roi):
	ret = []
	for channel in range(3):
		hist = cv.calcHist([roi], [channel], None, [256], [0,256])
		mu = histMean(hist)
		sigma = histDeviation(hist, mu)
		ret.append([mu,sigma])
	return ret
