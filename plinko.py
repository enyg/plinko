# Group 3 plinko project
# ECEn 631 winter 2019
import serial
import cv2

def setupGeometry(cap):
	# calibrate geometry for board position
	# (find edges, calculate pixels/cm, and calculate perspective transform, if necessary)
	cmPerPx = 3.0	# dummy value

def straighten(frame):
	# return straightened (maybe cropped) image of plinko board, based on initial calibration
	board = frame	# dummy function
	return board

def getCurrentBallPos(frame):
	# calculate x and y position in pixels for each ball (R,G,B)
	xr = 1	# dummy values
	yr = 2
	xg = 3
	yg = 4
	xb = 5
	yb = 6
	return [[xr, yr], [xg, yg], [xb, yb]]

# estimate the final horizontal position and time to reach the basket height (for each ball)
# return a value in cm for position
def estimateFinalBallPos(currPos):
	# rough guess for avg velocity in cm/sec - should be measured or estimated from video
	avgVel = 3.0
	# use current x positions as final x positions
	xrf = currPos[0][0] * cmPerPx
	xgf = currPos[1][0] * cmPerPx
	xbf = currPos[2][0] * cmPerPx
	# estimate final time based on avg velocity and current y position
	trf = (basketYPos - currPos[0][1])/avgVel
	tgf = (basketYPos - currPos[1][1])/avgVel
	tbf = (basketYPos - currPos[2][1])/avgVel
	
	return [[xrf, trf], [xbf, tbf], [xgf, tgf]]
	
def controlBasket(ballPrediction):
	# this function can move the motor at any time it is called,
	# or it can wait until the balls get close to the bottom, based on ball prediction (final x and time)
	
	# command = "g15"
	#ser.write((command +"\n").encode())
	return

#ser = serial.Serial('COM5', 115200, timeout=5)

cap = cv2.VideoCapture(0)

command = "h"

cmPerPx = 1/20.0	# dummy value - should be calibrated in setupGeometry()
basketYPos = 60.0	# dummy value - needs to be measured or set in calibration

if(cap.isOpened() == False):
	print("Error opening video file stream ior file")

# initialize plinko board geometry here
setupGeometry(cap)

while(cap.isOpened()):
	ret, frame = cap.read()

	cv2.imshow("video", frame)
	
	board = straighten(frame)
	
	ballPos = getCurrentBallPos(board)
	ballPrediction = estimateFinalBallPos(ballPos)
	controlBasket(ballPrediction)
	
	key = cv2.waitKey(10) & 0xFF

	if  key == ord('q'):
		break
	elif key == ord('i'):
		command = input("Enter Command")
		ser.write((command +"\n").encode())


