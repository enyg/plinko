# Group 3 plinko project
# ECEn 631 winter 2019
import serial
import cv2
import numpy as np


class Bot:

    def __init__(self, cap, ser):

        self.cap = cap
        if not self.cap.isOpened():
            raise ValueError("Error opening video file stream ior file")

        self.ser = ser
        self.home_command = 'h'
        self.pos_r = [-1,-1]  # dummy values
        self.pos_g = [-1,-1]
        self.pos_b = [-1,-1]
        self.ball_radius = 0.75 * 2.54 # ball radius in cms        
        # home basket
        #ser.write((self.home_commmand + "\n").encode())
        #ser.write(("g25\n").encode())
        self.lastBasketPosition = 25
        
        # initialize plinko board geometry here
        self.calibrate()

        # rough guess for avg velocity in cm/sec - should be measured or estimated from video
        self.avgVel = 3.0
                
    
    def calibrate(self):
        # (find edges, calculate pixels/cm, and calculate perspective transform, if necessary)
        # get bounding box for ROI
        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Calibration", 720, 480)
        cv2.setMouseCallback("Calibration", self.getCalibVerts)
        
        self.calibVerts = np.zeros([4,2], dtype="float32")
        self.vertIx = 0
        
        # display live feed while user clicks on points
        while True:
            success, self.calframe = cap.read()
            if not success:
                break
                
            # draw calibration vertices:
            for pt in self.calibVerts:
                if not all(pt == [0,0]):
                    cv2.circle(self.calframe, (pt[0],pt[1]), 4, [255,255,0], 1, cv2.LINE_AA)
            
            cv2.imshow("Calibration", self.calframe)
            cv2.waitKey(30)
            
            if self.vertIx >= 4:
                break
        
        ### get the transformation ###
        #(tl, tr, br, bl) = self.calibVerts
        #widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        #widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        #transW = max(int(widthA), int(widthB))
        
        #heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        #heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        #transH = max(int(heightA), int(heightB))
        
        # the board is 21" x 32"
        # these values will give 17 pixels per inch
        transW = 21*17  #357
        transH = 32*17  #544
        
        dst = np.array([
            [0, 0],
            [transW - 1, 0],
            [transW - 1, transH - 1],
            [0, transH - 1]], dtype = "float32")
        
        self.perspectiveTrans = cv2.getPerspectiveTransform(self.calibVerts, dst)
        self.transW = transW
        self.transH = transH
        
        # get cm per px scale
        # base it on known distance between screw holes in center of board
        # user will click 2 points for adjacent holes in the board
        
        # get center for region to do scale calibration with
        print("click region to use for scale calibration")
        # show the flattened/cropped image
        good = self.straighten(self.calframe)
        self.point = []
        cv2.setMouseCallback("Calibration", self.getPoint)
        cv2.imshow("Calibration",good)
        while self.point == []:
            cv2.waitKey(30)
        #cv2.waitKey(0)
        
        # show crop of area user chose
        width = good.shape[1]/3
        height = good.shape[0]/3
        cx = self.point[0]
        cy = self.point[1]
        x1 = max(0, round(cx - width/2))
        x2 = min(good.shape[0]-1, round(cx + width/2))
        y1 = max(0, round(cy - height/2))
        y2 = min(good.shape[1]-1, round(cy + height/2))
        #self.center = good[round(good.shape[1]/2 - width/2):round(good.shape[1]/2 + width/2), round(good.shape[0]/2 - height/2):round(good.shape[0]/2 + height/2)]
        
        # let user choose points for scale reference
        print("click on 2 points 1\" apart")
        self.scaleVerts = np.zeros([2,2], dtype="float32")
        self.vertIx = 0
        cv2.setMouseCallback("Calibration", self.getScaleVerts)
        while self.vertIx < 2:
            _, good = cap.read()
            good = self.straighten(good)
            self.center = good[x1:x2, y1:y2]
            for pt in self.scaleVerts:
                if not all(pt == [0,0]):
                    cv2.circle(self.center, (pt[0],pt[1]), 4, [255,255,0], 1, cv2.LINE_AA)
            cv2.imshow("Calibration",self.center)
            cv2.waitKey(100)
        
        knownDist = 2.54   # known distance in cm (grid holes are 1" apart)
        self.cmPerPx = knownDist/np.linalg.norm(self.scaleVerts[0]-self.scaleVerts[1])
        print("cm per pixel: ", self.cmPerPx)
        
        ### calibrate basket horizontal offset and height in px ###
        # move basket to known position
        #ser.write(("g25\n").encode())
        # user will click on top center of basket
        print("click on top center of basket")
        self.point = []
        cv2.setMouseCallback("Calibration", self.getPoint)
        while self.point == []:
            _, self.calframe = cap.read()
            good = self.straighten(self.calframe)
            cv2.imshow("Calibration",good)
            cv2.waitKey(30)
        
        # (x + offset) * cmPerPx = 25cm
        # offset = 25/cmPerPx - x
        # offset is also the x pixel value of the basket's 0cm position
        self.basketOffset = 25.0/self.cmPerPx - self.point[0]
        self.basketYPos = self.point[1]
        print("basket offset = ", self.basketOffset, " px")
        print("basket height = ", self.basketYPos, " px")
        
        cv2.destroyWindow("Calibration")
        
        # TODO: calibrate camera and undistort to make the board a true rectangle
    
    def getCalibVerts(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.calibVerts[self.vertIx] = [x,y]
            self.vertIx = self.vertIx + 1
            print("calibration pt: ", (x,y))
    
    def getPoint(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.point = [x,y]
            print((x,y))
    
    def getScaleVerts(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.scaleVerts[self.vertIx] = [x,y]
            #cv2.circle(self.center, (x,y), 4, [255,255,0], 1, cv2.LINE_AA)
            self.vertIx = self.vertIx + 1
            print("scale pt: ", (x,y))

    def straighten(self, frame):
        # return straightened (maybe cropped) image of plinko board, based on initial calibration
        return cv2.warpPerspective(frame, self.perspectiveTrans, (self.transW, self.transH))

    def getCurrentBallPos(self, frame):
        # calculate x and y position in pixels for each ball (R,G,B)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h = hsv[:,:,0]
        s = hsv[:,:,1]
        v = hsv[:,:,2]

        red_cond = ((h>=165)+(h<=15))*(s>100)* (v > 50)* (v < 200)
        green_cond = ((h>=45)*(h<=75))*(s>100)* (v > 50) * (v < 200)
        blue_cond = ((h>=105)*(h<=135))*(s>100)* (v > 50)* (v < 200)
        else_cond = ~(red_cond + green_cond+ blue_cond)

        frame[red_cond] = [0,0,255] # red 
        frame[green_cond] = [0,255,0] # green
        frame[blue_cond] = [255,0,0] # blue
        frame[else_cond] = [0,0,0]

        min_radius = int((self.ball_radius/self.cmPerPx)*0.75)
        max_radius = int((self.ball_radius/self.cmPerPx)*1.25)

        circle_red = cv2.HoughCircles(frame[:,:,2], cv2.HOUGH_GRADIENT, 1, minDist=20, param1=250, param2=8, minRadius=min_radius, maxRadius=max_radius)
        circle_green = cv2.HoughCircles(frame[:,:,1], cv2.HOUGH_GRADIENT, 1, minDist=20, param1=250, param2=8, minRadius=min_radius, maxRadius=max_radius)
        circle_blue = cv2.HoughCircles(frame[:,:,0], cv2.HOUGH_GRADIENT, 1, minDist=20, param1=250, param2=8, minRadius=min_radius, maxRadius=max_radius)

        ysize, xsize, channels = frame.shape
        x0 = (int)(3/self.cmPerPx)
        x1 = (int)(60/self.cmPerPx)
        y0 = (int)(10/self.cmPerPx)
        y1 = (int)(30/self.cmPerPx)
        y2 = (int)(80/self.cmPerPx)

        xy_diff = (int)(5/self.cmPerPx)

        if circle_red is not None:
            for x,y,r in circle_red[0,:]:

                if x0<x<x1 and y0<y<y2:

                    if self.pos_r == [-1,-1] and y<y1:
                        self.pos_r[0] = x
                        self.pos_r[1] = y

                    elif abs(self.pos_r[0]-x)<xy_diff and abs(self.pos_r[1]-y)<xy_diff:
                        self.pos_r[0] = x
                        self.pos_r[1] = y

        if circle_green is not None:
            for x,y,r in circle_green[0,:]:

                if x0<x<x1 and y0<y<y2:

                    if self.pos_g == [-1,-1] and y<y1:
                        self.pos_g[0] = x
                        self.pos_g[1] = y

                    elif abs(self.pos_g[0]-x)<xy_diff and abs(self.pos_g[1]-y)<xy_diff:
                        self.pos_g[0] = x
                        self.pos_g[1] = y

        if circle_blue is not None:
            for x,y,r in circle_blue[0,:]:

                if x0<x<x1 and y0<y<y2:

                    if self.pos_b == [-1,-1] and y<y1:
                        self.pos_b[0] = x
                        self.pos_b[1] = y

                    elif abs(self.pos_b[0]-x)<xy_diff and abs(self.pos_b[1]-y)<xy_diff:
                        self.pos_b[0] = x
                        self.pos_b[1] = y




        return [self.pos_r, self.pos_g, self.pos_b]

    # estimate the final horizontal position and time to reach the basket height (for each ball)
    # return a value in cm for position
    # this function translates from pixel values to cm on the basket's coordinate system
    def estimateFinalBallPos(self, currPos):
        [[xr, yr], [xg, yg], [xb, yb]] = currPos
        xfinal = [-1, -1, -1]
        tfinal = [-1, -1, -1]
        
        for ix in range(0,3):
            if currPos[ix][0] < 0:
                # this ball is not present (negative position in px).
                # Passing -1 as predictions will indicate to control function
                # that this ball is not on the board anymore.
                xfinal[ix] = -1
                tfinal[ix] = -1
            else:
                # final x position is just set to current position (transformed)
                xfinal[ix] = (currPos[ix][0] + self.basketOffset) * self.cmPerPx
                # estimate final time based on avg velocity and current y position
                tfinal[ix] = (self.basketYPos - currPos[ix][1])/self.avgVel

        #return [[xrf, trf], [xgf, tgf], [xbf, tbf]]
        return xfinal, tfinal

    def controlBasket(self, xfinal, tfinal):
        # this function can move the basket at any time it is called,
        # or it can wait until the balls get close to the bottom, based on ball prediction (final x and time)
        [xrf, xgf, xbf] = xfinal
        [trf, tgf, tbf] = tfinal
        
        # track red ball until it is past the basket height
        if trf > 0:
            newPos = xrf
        else:
            if tgf <= 0:
                newPos = xbf
            elif tbf <= 0:
                newPos = xgf
            elif np.abs(xgf - self.lastBasketPosition) < np.abs(xbf - self.lastBasketPosition):
                newPos = xgf
            else:
                newPos = xbf

        #ser.write(("g" + str(round(newPos)) + "\n").encode()
        self.lastBasketPosition = round(newPos)
        
        # command = "g15"
        # ser.write((command +"\n").encode())
        return

    def run(self):
        cv2.namedWindow("video", cv2.WINDOW_NORMAL)
        while self.cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            board = self.straighten(frame)
            cv2.imshow("video", board)
            ballPos = self.getCurrentBallPos(board)
            xfinal, tfinal = self.estimateFinalBallPos(ballPos)
            self.controlBasket(xfinal, tfinal)

            key = cv2.waitKey(10) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('i'):
                command = input("Enter Command")
                self.ser.write((command + "\n").encode())


#cap = cv2.VideoCapture(2)
cap = cv2.VideoCapture("sample.avi")
# 800 x 448 works with 24 fps
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 448)
# ser = serial.Serial('COM5', 115200, timeout=5)
ser = None

bot = Bot(cap=cap, ser=ser)
bot.run()
