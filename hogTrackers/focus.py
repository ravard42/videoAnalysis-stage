import cv2 as cv
import numpy as np
from skimage.feature import hog
from skimage import color
import math as m

corn = 18
hogCorn = 9
quality = 0.03
minDist = 0.1

old = (5,11,17)


bins = 9
#on cherche la distribution autour du corner donc nombre impair pour la taille des cellules
C = 5
cellSize = (C,C)
B = 5
cellPBlock = (B,B)
Z = (C * B - 1) / 2

queryErr = 0

def computeHog(kPs, frame):
	des = []
	for p in kPs:
		x,y = p.ravel()
		X, Y = int(x), int(y)
		roi = frame[Y-Z:Y+Z+1,X-Z:X+Z+1]
		gray_roi = color.rgb2gray(roi)
		des.append(hog(gray_roi, orientations=bins,
			pixels_per_cell=cellSize, cells_per_block=cellPBlock, block_norm='L2-Hys'))
	des = np.array(des)
	des = des.astype(np.float32)
	return des

def printResults(matches):
	print "<-----START------>"
	for elem in matches:
		i = 0
		print "<----------->"
		while i < hogCorn:
			qId = elem[i].queryIdx
			tId = elem[i].trainIdx
			print "(k={})-> {}:{}".format(i,qId,tId)
			i += 1

def selectResults(kPs ,matches):
	#kPs est une copie du env[6] INITIAL: 
	#Il est consituE des shiTomasi les plus "purs" de la fenetre de focus de la frame courante
	#On doit en extraire EXACTEMENT 5,
	#5 hogTrackers en coherence optimale (hogDescriptor) avec les trainKPs(env[8]) de la selection a la souris
	#ces 5 points constituront le retour de la fonction que l'on nommera env[6] OPTIMISE (pour fixer les idees)

	ret = np.array([[[-42,-42]] for i in range(hogCorn)])
	tab = []
	for elem in matches:
		row = [elem[i] for i in range(hogCorn)]
		tab.append(row)
	n = len(tab)
	poubelle = [-42 for i in range(n)]
		#NB: les indices du env[6] INITIAL deja attribuEs au env[6] OPTIMISE
	k = 0
	while k < hogCorn:
		i = 0
		tmp = [-42 for w in range(hogCorn)]
		while i < n:
			if poubelle[i] != -42:
				i += 1
				continue
			trId = tab[i][k].trainIdx
			d = tab[i][k].distance
			if ret[trId][0][0] == -42:
				ret[trId][0][0] = kPs[i][0][0]
				ret[trId][0][1] = kPs[i][0][1]
				tmp[trId] = d
				poubelle[i] = trId
			elif d < tmp[trId]:
				ret[trId][0][0] = kPs[i][0][0]
				ret[trId][0][1] = kPs[i][0][1]
				tmp[trId] = d
				for p,q in enumerate(poubelle):
					if q == trId:
						poubelle[p] = -42
				poubelle[i] = trId
			i +=1
		k += 1
#	print "<----env[6] INITIAL-----> \n {}".format(kPs)
#	print"<=====POUBELLE====> \n {}".format(poubelle)
#	print "<----env[6] OPTIMISE-----> \n {}".format(ret)
	return ret	
	
def drawHogTrackCorn((x,y),i,env):
	if i < 3:
		cv.circle(env[4],(x,y),1,(0,0,255),-1)
	elif i < 6:
		cv.circle(env[4],(x,y),1,(0,255,0),-1)
	else:
		cv.circle(env[4],(x,y),1,(255,0,0),-1)

def forward(env):
	global queryErr

	env[5] = np.zeros(env[4].shape[:2], np.uint8)
	env[5][env[1]-env[3]:env[1]+env[3], env[0]-env[2]:env[0]+env[2]] = 255
	env[6] = cv.goodFeaturesToTrack(cv.cvtColor(env[4], cv.COLOR_BGR2GRAY),corn,quality,minDist, mask=env[5])
	if env[6] is not None and len(env[6]) >= hogCorn:
		queryErr = 0
		env[7] = computeHog(env[6], env[4])
		bf = cv.BFMatcher()
		matches = bf.knnMatch(env[7], env[9], k=hogCorn)
		#printResults(matches)
		env[6] = selectResults(env[6], matches)
		i = 0
		for p in env[6]:
			x, y = p.ravel()
			drawHogTrackCorn((x,y),i,env)
			i += 1
		env[0] = int(sum(env[6][:,...,0].ravel()) / len(env[6]))
		env[1] = int(sum(env[6][:,...,1].ravel()) / len(env[6]))
	else:
		if queryErr == 0:
			print "pas assez de queryKps"
		queryErr = 1

def refreshTrainKPs(env, i):
	if i%old[0] == 0:
		serie = 0
	if i%old[1] == 0:
		serie = 1
	if i%old[2] == 0:
		serie = 2

	env[5] = np.zeros(env[4].shape[:2], np.uint8)
	env[5][env[1]-env[3]:env[1]+env[3], env[0]-env[2]:env[0]+env[2]] = 255
	env[10] = cv.goodFeaturesToTrack(cv.cvtColor(env[4], cv.COLOR_BGR2GRAY),hogCorn / 3,quality,minDist, mask=env[5])
	if env[10] is not None and len(env[10]) == hogCorn / 3:
		env[11] = computeHog(env[10], env[4])
		while i < hogCorn / 3:
			env[8][serie * hogCorn / 3 + i] = env[10][i]
			env[9][serie * hogCorn / 3 + i] = env[11][i]
			i += 1
	else:
		print "pas assez de trainKPs, reessayez"
		env[8] = None

def foc(event, x, y, flags, env):
	if event == cv.EVENT_FLAG_LBUTTON:
		env[0], env[1] = x, y
		env[5] = np.zeros(env[4].shape[:2], np.uint8)
		env[5][env[1]-env[3]:env[1]+env[3], env[0]-env[2]:env[0]+env[2]] = 255
		env[8] = cv.goodFeaturesToTrack(cv.cvtColor(env[4], cv.COLOR_BGR2GRAY),hogCorn,quality,minDist, mask=env[5])
		if env[8] is not None and len(env[8]) == hogCorn:
			env[9] = computeHog(env[8], env[4])
		else:
			print "pas assez de trainKPs, reessayez"
			env[8] = None
