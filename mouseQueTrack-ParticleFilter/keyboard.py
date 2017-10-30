pas = 8
coef = 4

def printEnvInfos(env):
	print "(x,y) = ({},{})".format(env[0], env[1])
	print "(w/2,h/2) = ({},{})".format(env[2], env[3])
	nbPix = (env[2] * 2 - 1)*(env[3] * 2 - 1)
	print "nbPixel in hist : {}".format(nbPix)
	if env[5]:
		print "<----ENV5----->"
		for i in range(3):
			print "	CANAL {}".format(i)
			print "		----> mu = {}".format(env[5][i][0])
			print "		----> sigma = {}".format(env[5][i][1])
		if len(env[6]) > 0:
			print "<----ENV6----->"
			print env[6]

def loop(k, env):
	global pas, coef
	if k == ord('q'):
		return 0
	else:
		if k == 57 and (pas == 8 or pas == coef * 8):
				pas = coef * 8 if pas == 8 else 8
		if k == 55 and (pas == 8 or pas == 8 / coef):
				pas = 8 / coef if pas == 8 else 8
		if k == 54:
			env[2] += pas
		if k == 52 and env[2] > pas:
			env[2] -= pas
		if k == 56:
			env[3] += pas
		if k == 50 and env[3] > pas:
			env[3] -= pas
		if k == ord('i'):
			printEnvInfos(env)
		if k == ord('h'):
			env[5] = None
		if k == ord('p'):
			env[6] = []
		return 1
