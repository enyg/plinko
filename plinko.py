
import serial
import cv2

ser = serial.Serial('COM5', 115200, timeout=5)


cap = cv2.VideoCapture(0)

command = "h"

if(cap.isOpened() == False):
	print("Error opening video file stream ior file")



while(cap.isOpened()):
	ret, frame = cap.read()

	cv2.imshow("video", frame)
	key = cv2.waitKey(10) & 0xFF

	if  key == ord('q'):
		break
	elif key == ord('i'):
		command = input("Enter Command")
		ser.write((command +"\n").encode())
