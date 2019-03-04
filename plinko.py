# Group 3 plinko project
# ECEn 631 winter 2019
import serial
import cv2
import numpy as np
import time


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
        self.found = [False,False,False]
        self.ball_radius = 0.75 * 2.54 # ball radius in cms        
        # home basket
        ser.write(('h'+ "\n").encode())
        ser.write(("g25\n").encode())
        self.lastBasketPosition = 25
        
        # initialize plinko board geometry here
        self.calibrate()

        # rough guess for avg velocity in cm/sec - should be measured or estimated from video
        self.avgVel = 11.0

        self.basket_velocity = 36    # 25 cm/sec
        # Red (5 points), Green (4 points), and Blue (3 points)
        self.scores = np.array([5, 4, 3])

        self.combination = np.zeros((8, 3), dtype=np.int)
        for i in range(3):
            self.combination[i+1, i] = 1
            self.combination[i+4, i] = 1
        self.combination[4:, :] = 1 - self.combination[4:, :]

    # set some guessed calibration values 
    # (for testing purposes when you don't want to do a full calibration)
    def calibrateLazy(self):
        #self.calibVerts = np.array([[4,1],[356,3],[354,525],[3,520]], dtype="float32")
        self.calibVerts = np.array([[716,51],[711,415],[164,392],[179,38]], dtype="float32")
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
        
        #self.cmPerPx = 0.149
        #self.basketOffset = -3.677
        #self.basketYPos = 525
        
        self.cmPerPx = 0.1411
        self.basketOffset = 30.165
        self.basketYPos = 527
    
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
            success, self.calframe = self.cap.read()
            if not success:
                print('not reading')
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
        
        print("dimensions: ", transW, transH)
        
        ### get cm per px scale ###
        # base it on known distance between screw holes in center of board
        # user will click 2 points for adjacent holes in the board
        
        # get center for region to do scale calibration with
        #print("click region to use for scale calibration")
        # show the flattened/cropped image
        good = self.straighten(self.calframe)
        #self.point = []
        #cv2.setMouseCallback("Calibration", self.getPoint)
        #cv2.imshow("Calibration",good)
        #while self.point == []:
        #    cv2.waitKey(30)
        #cv2.waitKey(0)
        
        # show crop of area user chose
        #width = good.shape[0]/3
        #height = good.shape[1]/3
        #cx = self.point[0]
        #cy = self.point[1]
        #x1 = max(0, round(cx - width/2))
        #x2 = min(good.shape[0]-1, round(cx + width/2))
        #y1 = max(0, round(cy - height/2))
        #y2 = min(good.shape[1]-1, round(cy + height/2))
        #self.center = good[round(good.shape[1]/2 - width/2):round(good.shape[1]/2 + width/2), round(good.shape[0]/2 - height/2):round(good.shape[0]/2 + height/2)]
        #print(cx, cy, good.shape)
        #print(x1, x2, "\n", y1, y2)
        
        # let user choose points for scale reference
        #print("click on 2 points 1\" apart")
        #self.scaleVerts = np.zeros([2,2], dtype="float32")
        #self.vertIx = 0
        #cv2.setMouseCallback("Calibration", self.getScaleVerts)
        #while self.vertIx < 2:
        #    _, good = cap.read()
        #    good = self.straighten(good)
        #    self.center = good[x1:x2, y1:y2]            
        #    for pt in self.scaleVerts:
        #        if not all(pt == [0,0]):
        #            cv2.circle(self.center, (pt[0],pt[1]), 4, [255,255,0], 1, cv2.LINE_AA)
        #    cv2.imshow("Calibration",self.center)
        #    cv2.waitKey(100)
        
        #knownDist = 2.54   # known distance in cm (grid holes are 1" apart)
        #self.cmPerPx = knownDist/np.linalg.norm(self.scaleVerts[0]-self.scaleVerts[1])
        self.cmPerPx = 21*2.54/good.shape[1]
        print("cm per pixel: ", self.cmPerPx)
        
        ### calibrate basket horizontal offset and height in px ###
        # move basket to known position
        ser.write(("g25\n").encode())
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
            # self.scaleVerts[self.vertIx] = [x,y]
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

        red_cond = ((h>=165)+(h<=15))*(s>80)* (v > 80)* (v < 215)
        green_cond = ((h>=45)*(h<=75))*(s>100)* (v > 80) * (v < 215)
        blue_cond = ((h>=105)*(h<=135))*(s>100)* (v > 80)* (v < 215)
        else_cond = ~(red_cond + green_cond+ blue_cond)

        frame1 = frame.copy()

        frame1[red_cond] = [0,0,255] # red 
        frame1[green_cond] = [0,255,0] # green
        frame1[blue_cond] = [255,0,0] # blue
        frame1[else_cond] = [0,0,0]

        min_radius = int((self.ball_radius/self.cmPerPx)*0.75)
        max_radius = int((self.ball_radius/self.cmPerPx)*1.25)

        circle_red = cv2.HoughCircles(frame1[:,:,2], cv2.HOUGH_GRADIENT, 1, minDist=20, param1=250, param2=10, minRadius=min_radius, maxRadius=max_radius)
        circle_green = cv2.HoughCircles(frame1[:,:,1], cv2.HOUGH_GRADIENT, 1, minDist=20, param1=250, param2=10, minRadius=min_radius, maxRadius=max_radius)
        circle_blue = cv2.HoughCircles(frame1[:,:,0], cv2.HOUGH_GRADIENT, 1, minDist=20, param1=250, param2=10, minRadius=min_radius, maxRadius=max_radius)

        ysize, xsize, channels = frame.shape
        x0 = (int)(2/self.cmPerPx)
        x1 = (int)(65/self.cmPerPx)
        y0 = (int)(10/self.cmPerPx)
        y1 = (int)(25/self.cmPerPx)
        y2 = (int)(75/self.cmPerPx)
        #print(x0,x1,y0,y1,y2)
        xy_diff = (int)(15/self.cmPerPx)

        self.found = [False,False,False]

        if circle_red is not None:
            for x,y,r in circle_red[0,:]:

                if x0<x<x1 and y0<y:
                    
                    if self.pos_r == [-1,-1] and y<y1:
                        self.pos_r[0] = x
                        self.pos_r[1] = y
                        self.found[0] = True
                        #cv2.circle(frame1,(x,y),int(r*2),(0,0,255),-1)

                    elif abs(self.pos_r[0]-x)<xy_diff and -10<(y-self.pos_r[1])<xy_diff:

                        if y<y2:
                            self.pos_r[0] = x
                            self.pos_r[1] = y
                            self.found[0] = True
                        else:
                            self.pos_r[0] = -1
                            self.pos_r[1] = -1
                    break



        if circle_green is not None:
            for x,y,r in circle_green[0,:]:

                if x0<x<x1 and y0<y:

                    if self.pos_g == [-1,-1] and y<y1:
                        self.pos_g[0] = x
                        self.pos_g[1] = y
                        self.found[1] = True
                        #cv2.circle(frame1,(x,y),int(r*2),(0,255,0),-1)

                    elif abs(self.pos_g[0]-x)<xy_diff and -10<(y-self.pos_g[1])<xy_diff:
                        
                        if y<y2:
                            self.pos_g[0] = x
                            self.pos_g[1] = y
                            self.found[1] = True
                        else:
                            self.pos_g[0] = -1
                            self.pos_g[1] = -1
                    break

        if circle_blue is not None:
            for x,y,r in circle_blue[0,:]:

                if x0<x<x1 and y0<y:

                    if self.pos_b == [-1,-1] and y<y1:
                        self.pos_b[0] = x
                        self.pos_b[1] = y
                        self.found[2] = True
                        #cv2.circle(frame1,(x,y),int(r*2),(255,0,0),-1)

                    elif abs(self.pos_b[0]-x)<xy_diff and -10<(y-self.pos_b[1])<xy_diff:
                        if y<y2:
                            self.pos_b[0] = x
                            self.pos_b[1] = y
                            self.found[2] = True
                        else:
                            self.pos_b[0] = -1
                            self.pos_b[1] = -1
                    break


        cv2.imshow('circles detected', frame1)
        #print([self.pos_r, self.pos_g, self.pos_b])

        return [self.pos_r, self.pos_g, self.pos_b]

    # estimate the final horizontal position and time to reach the basket height (for each ball)
    # return a value in cm for position
    # this function translates from pixel values to cm on the basket's coordinate system
    def estimateFinalBallPos(self, currPos, img, draw):
        xfinal = [-1, -1, -1]
        tfinal = [-1, -1, -1]
        
        pxPerSec = self.avgVel / self.cmPerPx
        
        for ix in range(0,3):
            if self.found[ix] == False:
                if ix == 0 and self.pos_r[1] != -1:
                    self.pos_r[1] = self.pos_r[1] + pxPerSec * self.dt
                elif ix == 1 and self.pos_g[1] != -1:
                    self.pos_g[1] = self.pos_g[1] + pxPerSec * self.dt
                elif ix == 2 and self.pos_b[1] != -1:
                    self.pos_b[1] = self.pos_b[1] + pxPerSec * self.dt
        
        print("red: ", self.pos_r, self.found[0], self.dt, "s")
        
        [[xr, yr], [xg, yg], [xb, yb]] = [self.pos_r, self.pos_g, self.pos_b]   #currPos
        
        for ix in range(0,3):
            # set ball pos to -1 if past basket
            if currPos[ix][1] > self.basketYPos:
                if ix ==0:
                    self.pos_r = [-1,-1]
                elif ix == 1:    
                    self.pos_g = [-1,-1]
                elif ix == 2:
                    self.pos_b = [-1,-1]
            
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
        
        # draw predicted path on the board image (if draw = True)
        # this part works with pixel values, not centimeters - conversion is only done for the final prediction
        dt = self.dt    # use last delta t as assumption for frame rate
        #drawstart = time.perf_counter()
        if draw == True:
            # index 0: r/g/b   index 1: x/y    index 2: time step
            timeSteps = int(max(tfinal)/dt)
            if timeSteps > 0:
                paths = np.zeros([3, 2, timeSteps], dtype="float32")
                for ix in range(0,3):
                    if tfinal[ix] >= 0:
                        color = (255*(ix==2), 255*(ix==1), 255*(ix==0))
                        paths[ix][0][:] = np.ones([timeSteps])*currPos[ix][0]
                        for tix in range(0, int(tfinal[ix]/dt)):
                            paths[ix][1][tix] = currPos[ix][1]+pxPerSec*dt*tix
                        for tix in range(0, int(tfinal[ix]/dt)):                    
                            cv2.circle(img, (paths[ix][0][tix], paths[ix][1][tix]), 4, color, cv2.FILLED)
        #drawend = time.perf_counter()
        #print("drawing: ", drawend-drawstart, " s")
        
        #return [[xrf, trf], [xgf, tgf], [xbf, tbf]]
        return xfinal, tfinal

    def controlBasket(self, xfinal, tfinal):
        # this function can move the basket at any time it is called,
        # or it can wait until the balls get close to the bottom, based on ball prediction (final x and time)
        xfinal = np.array(xfinal)
        tfinal = np.array(tfinal)

        scores_sorted = np.empty_like(self.scores)


        # start processing
        access = np.ones(8)

        dt_move = np.zeros((3, 3))
        dt_fall = np.zeros((3, 3))

        idx_sorted = np.argsort(tfinal)
        for i in range(3):
            scores_sorted[i] = self.scores[idx_sorted[i]]
            for j in range(3):
                if i == j:
                    break
                dt_move[i, j] = np.abs(xfinal[idx_sorted[i]] - xfinal[idx_sorted[j]])
                dt_move[j, i] = dt_move[i, j]
                dt_fall[i, j] = np.abs(tfinal[idx_sorted[i]] - tfinal[idx_sorted[j]])
                dt_fall[j, i] = dt_fall[i, j]

        scoreboard = self.combination * scores_sorted
        scoreboard = scoreboard.sum(axis=1)
        dt_move /= self.basket_velocity

        # compare time interval
        enough_time = (dt_move <= dt_fall)

        # block impossible combinations
        if not enough_time[0, 1]:
            access[6] = 0
            access[7] = 0
        if not enough_time[1, 2]:
            access[4] = 0
            access[7] = 0
        if not enough_time[0, 2]:
            access[5] = 0

        # choose best strategy from possible combinations
        combination_verified = self.combination[access > 0]
        scoreboard_verified = scoreboard[access > 0]
        idx_max_score = scoreboard_verified.argmax()
        decision = combination_verified[idx_max_score]

        # show the strategy
        _map = {0: "Unselect", 1: "Select"}
        print("1st ball: {}\n2nd ball: {}\n3rd ball: {}".format(
            _map[decision[0]], _map[decision[1]], _map[decision[2]]))

        for i in range(3):
            if tfinal[idx_sorted[i]] > 0:
                if decision[i] != 0:
                    newPos = xfinal[idx_sorted[i]]
                    #print("basket: ", int(round(newPos)), " cm")
                    ser.write(("g" + str(int(round(newPos))) + "\n").encode())
                    self.lastBasketPosition = round(newPos)
                    break
        
        # command = "g15"
        #ser.write((command +"\n").encode())
        return

    def run(self):
        self.t = time.perf_counter()  # current frame time
        self.dt = 0 # difference from last frame time
        
        cv2.namedWindow("video")#, cv2.WINDOW_NORMAL)
        while self.cap.isOpened():
            success, frame = cap.read()
            # update times
            tnew = time.perf_counter()
            self.dt = tnew - self.t
            self.t = tnew
            if not success:
                break

            board = self.straighten(frame)
            #cv2.imshow("video", board)
            ballPos = self.getCurrentBallPos(board)
            xfinal, tfinal = self.estimateFinalBallPos(ballPos, board, draw=True)
            cv2.imshow("video", board)
            self.controlBasket(xfinal, tfinal)

            key = cv2.waitKey(10) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('i'):
                command = input("Enter Command")
                self.ser.write((command + "\n").encode())
            elif key == ord('s'):
                # manually reset ball states to all not present
                self.pos_r = [-1,-1]  
                self.pos_g = [-1,-1]
                self.pos_b = [-1,-1]

ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=5)
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(1)#"sample.avi")
# 800 x 448 works with 24 fps
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 448)

time.sleep(2)

#ser = None

bot = Bot(cap=cap, ser=ser)
bot.run()
