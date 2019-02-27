# Group 3 plinko project
# ECEn 631 winter 2019
import serial
import cv2


class Bot:

    def __init__(self, cap, ser):

        self.cap = cap
        if not self.cap.isOpened():
            raise ValueError("Error opening video file stream ior file")

        self.ser = ser
        self.home_command = 'h'

        self.cmPerPx = 1/20.0  # dummy value - should be calibrated in setupGeometry()
        self.basketYPos = 60.0  # dummy value - needs to be measured or set in calibration

        # initialize plinko board geometry here
        # (find edges, calculate pixels/cm, and calculate perspective transform, if necessary)
        self.cmPerPx = 3.0  # dummy value

        # rough guess for avg velocity in cm/sec - should be measured or estimated from video
        self.avgVel = 3.0

    def straighten(self, frame):
        # return straightened (maybe cropped) image of plinko board, based on initial calibration
        board = frame  # dummy function
        return board

    def getCurrentBallPos(self, frame):
        # calculate x and y position in pixels for each ball (R,G,B)
        xr = 1  # dummy values
        yr = 2
        xg = 3
        yg = 4
        xb = 5
        yb = 6
        return [[xr, yr], [xg, yg], [xb, yb]]

    # estimate the final horizontal position and time to reach the basket height (for each ball)
    # return a value in cm for position
    def estimateFinalBallPos(self, currPos):

        # use current x positions as final x positions
        xrf = currPos[0][0] * self.cmPerPx
        xgf = currPos[1][0] * self.cmPerPx
        xbf = currPos[2][0] * self.cmPerPx
        # estimate final time based on avg velocity and current y position
        trf = (self.basketYPos - currPos[0][1])/self.avgVel
        tgf = (self.basketYPos - currPos[1][1])/self.avgVel
        tbf = (self.basketYPos - currPos[2][1])/self.avgVel

        return [[xrf, trf], [xbf, tbf], [xgf, tgf]]

    def controlBasket(self, ballPrediction):
        # this function can move the ser at any time it is called,
        # or it can wait until the balls get close to the bottom, based on ball prediction (final x and time)

        # command = "g15"
        # ser.write((command +"\n").encode())
        return

    def run(self):

        while self.cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            cv2.imshow("video", frame)

            board = self.straighten(frame)
            ballPos = self.getCurrentBallPos(board)
            ballPrediction = self.estimateFinalBallPos(ballPos)
            self.controlBasket(ballPrediction)

            key = cv2.waitKey(10) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('i'):
                command = input("Enter Command")
                self.ser.write((command + "\n").encode())


cap = cv2.VideoCapture(0)
# ser = serial.Serial('COM5', 115200, timeout=5)
ser = None

bot = Bot(cap=cap, ser=ser)
bot.run()
