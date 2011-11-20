from PIL import Image
from numpy import *
from math import *
image = Image.open('TokyoPanoramaShredded.png')
imageDx, imageDy = image.size 

def main():		
	stripDx, noOfStrips = stripDxCalculator()
	box = [0, 0, stripDx, imageDy] #  defines coordinates NE and SW corners of strips
	strip = []   # list of shredded strips of the source image
	for i in range(noOfStrips):
		box[0:4:2] = [stripDx * i, stripDx * (i + 1)] # set x-coors
		strip.append(image.crop(box))
	unshuffler(strip, noOfStrips).show()
	unshuffler(strip, noOfStrips).save("TokyoPanoramaDeShuffled.png")
	
def stripDxCalculator():
	global R, stripDx
	R = getRGB(0)
	stripDx = imageDx / 2
	mismatch = zeros(imageDx / 2)
	total = 0   # to be used for "cumulativeAve"
	violationNo = 0 
	for x in range(imageDx / 2 + 2):
		mismatch[x] = mismatchCalc(R, 0, x, 0, x + 1, h=5)
		total += mismatch[x]
		cumulativeAve = total / (x + 1)
		if mismatch[x] > 1.5 * cumulativeAve: 
			violationNo += 1
		if violationNo == 3:   # 3 is a design value. Could be larger to be more confident
			index = nonzero(mismatch > 1.5 * cumulativeAve)
			break
	index = index[0]
	stripDx = diff(index).min()   # (allow original strips to end up next to each other)
	return stripDx, imageDx / stripDx
	
def unshuffler(strip, noOfStrips):
	imageUnshuffled = Image.new(image.mode, image.size) 
	mismatch = zeros(noOfStrips)
	min = zeros(noOfStrips)
	rightNeighbor = zeros((noOfStrips, 2), dtype = int16)
	sequence = zeros(noOfStrips, dtype = int16)
	for i in range (noOfStrips):
		for k in range(noOfStrips): # compare i-th strip on Left, k-th strip on Right
			mismatch[k] = mismatchCalc(R, i, stripDx - 1, k, 0, h=5) 
		min[i] = mismatch.min() # small mismatch value means the interface matches well.
		rightNeighbor[i][0] = i 
		rightNeighbor[i][1] = mismatch.argmin() 
		# Locate rightNeighbor[i][0]-th strip to the left of rightNeighbor[i][1]-th strip
	Right = min.argmax()
	rightNeighbor[Right][1] = -1 
	# sequence: array of strip labels in the deshuffled image.
	sequence[noOfStrips - 1] = Right
	imageUnshuffled.paste(strip[Right], (stripDx * (noOfStrips - 1), 0))
	for i in range(noOfStrips - 1 - 1, -1, -1):
		sequence[i] = nonzero(rightNeighbor[:, 1] == sequence[i + 1])[0]
		imageUnshuffled.paste(strip[sequence[i]], (stripDx * i, 0))
	return imageUnshuffled

def getRGB(color):
	R = zeros((imageDx, imageDy), dtype = int16)
	for x in range(imageDx):  
		for y in range(imageDy):  
			R[x][y] = get_pixel_value(x, y)[color]
	return R

def get_pixel_value(x, y):
	data = image.getdata() 
	pixel = data[y * imageDx + x]
	return pixel

def xg(i, x): 
	# returns global x-coor
	global stripDx
	return x + (i * stripDx)

def mismatchCalc(R, i, xi, k, xk, h): 
	# i-th strip on Left, k-th strip on Right. 
	# xi, xk: local x-coors of i-th and k-th strips. 
	# h: Averaging window height. (must be >1, around 3-5 works well)
	global stripDx	
	thetaThreshold = 10.0 / (180.0 / pi) # Should be around (10,20) degrees
	tape = zeros(imageDy, dtype = int8)
	for y in range(h / 2, imageDy - (h / 2) - 1):
		yb = y - (h / 2)
		yt = y + (h / 2)
		gradx =(float(sum(R[xg(k, xk), yb:yt + 1])) - 
				float(sum(R[xg(i, xi), yb:yt + 1]))
				) / h
		grady = float(R[xg(i, xi)][yt] - R[xg(i, xi)][yb]) / h
		if gradx == 0.0:
			theta = pi / 2.0 # angle btw the grad vector and pos x-axis. (ccw +)
		else:
			theta = (atan(grady / gradx))
		if fabs(theta - 0.0) < thetaThreshold:
			tape[y] = 1  # tape[y] = 1 => mismatching interface, 0 otherwise.
	mismatch = float(sum(tape)) / (imageDy - 2*(h / 2) - 1)   # a ratio in (0,1)
	return mismatch
	