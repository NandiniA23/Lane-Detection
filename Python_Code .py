##################################IMPORTING ESSENTIAL LIBRARIES##################################
import cv2                                                                                                              					#For image processing.
import numpy as np                                                                                                      					#For matrice operations.
import math 											#For converting radians to degrees.
##########################################INITIALISATION###########################################
def make_points(image, line): 										#Function which will give the desired end point coordinates of a line, given its slope and y-intercept.
    slope, intercept = line 										#Extracting the slope and intercept from the line list.
    y1 = int(image.shape[0])										#Bottom of the image
    y2 = int(y1*3/5)         										#Slightly lower than the middle
    x1 = int((y1 - intercept)/slope) 									#x coordinate at the point (x1,y1)
    x2 = int((y2 - intercept)/slope) 									#x coordinate at the point (x2,y2)
    return [[x1, y1, x2, y2]] 										#Return the new line coordinates
												#
cap=cv2.VideoCapture('Videos/LANE_VIDEO.mp4')                                                                   				#Reading the video file
while(True):											#Infinite loop.
    ret,frame=cap.read()                                                                                                					#Read the next frame of the video.
    if (ret): 												#When ret is False, the video is over.
        line_image = frame.copy() 									#Create a clone of the frame (This will be untouched).
        original = frame.copy() 										#Create a clone of the frame (To draw unprocessed lines).
        height=frame.shape[0]                                                                                              				 	#Find height(no.of rows of image).
        width=frame.shape[1]                                                                                               				 	#Find width(no.of columns in the image).
        roi_vertices=[(0,height),(width/2+55,height/2+50),(width*2/3-55,height/2+50),(width,height-200),(width,height)] 	                  #Set the region of interest (i.e. road).
        #cv2.imshow("ROI",roi_vertices)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)                                                                         			#Convert BGR to grayscale image
        canny_img=cv2.Canny(gray,50,150)                                                                                   				#Apply Canny edge detection with minVal 100 and maxVal 200
        mask=np.zeros_like(canny_img)                                                                                       				#Create a zeros image of same dimension as the edge detected image
        cv2.fillPoly(mask,np.array([roi_vertices],np.int32),255)                                                            				#The polygonal area bound by the specified vertices is filled with 255(i.e. white)
        cropped_img=cv2.bitwise_and(canny_img,mask)                                                                         			#AND operation performed between edge detected image and mask to crop the image
        lines=cv2.HoughLinesP(cropped_img,rho=2,theta=np.pi/180,threshold=100,minLineLength=40,maxLineGap=5)               	#Apply probabilistic Hough Transform on the cropped edge-detected image
        left_fit = [] 											#Array to store left lane lines' slopes and intercepts.
        right_fit = [] 											#Array to store right lane lines' slopes and intercepts.
        for line in lines:                                                                                                  					#Iterating on each of the lines detected
           if line is None: 										#if no lines were detected,
               pass 											#skip this iteration.
           else: 											#If lines were detected.
               x1,y1,x2,y2=line[0]                                                                                              					#Find out the starting and ending points of the line segment
               cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),2) 							#Draw a line joining the end points.
               fit = np.polyfit((x1,x2), (y1,y2), 1) 									#Find the slope and intercept of a line given the end points.
               slope = fit[0] 										#Get the slope.
               intercept = fit[1] 										#Get the intercept.
               m = abs(slope) 										#Magnitude of the slope.
               slope_d = math.degrees(math.atan(m)) 								#Finding the slope angle in terms of degrees.
               if slope_d >=5: 										#If the slope angle is above 5 degrees.
                   if slope < 0: 										#If the actual slope is -ve.
                       left_fit.append((slope, intercept)) 								#Store the line details in the left fit array.
                   else:											#If the slope is positive.
                       right_fit.append((slope, intercept)) 								#Store the line details in the right fit array.
        left_fit_average  = np.average(left_fit, axis=0) 								#Find the average slope and intercept value from all the left lines.
        right_fit_average = np.average(right_fit, axis=0) 							#Find the average slope and intercept value from all the right lines.
        if len(left_fit) != 0 and len(right_fit) != 0: 								#If atleast one left and right line were detected.
            left_line = make_points(frame, left_fit_average) 							#Find the end points of the left line using make_points function.
            right_line = make_points(frame, right_fit_average) 							#Find the end points of the right line using make_points function.
            lx1,ly1,lx2,ly2 = left_line[0] 									#Extract the end points.
            rx1,ry1,rx2,ry2 = right_line[0] 									#Extract the end points.
            cv2.line(frame,(lx1,ly1),(lx2,ly2),(255,0,0),2) 								#Draw the processed left line.
            cv2.line(frame,(rx1,ry1),(rx2,ry2),(255,0,0),2) 								#Draw the processed right line.
        cv2.imshow("Lane_detection",frame)                                                                                           			#Display the Images.
        cv2.imshow("Lines",line_image) 									#
        cv2.imshow("POLYGONAL ROI",cropped_img) 							#
        cv2.imshow("POLYGONAL ROI MASK",mask) 								#
        cv2.imshow("CANNY_EDGE",canny_img) 								#
        cv2.imshow("GRAYSCALE",gray) 									#
        cv2.imshow("ORIGINAL",original) 									#
												#
        if cv2.waitKey(1) & 0xFF==ord('q'):                                                                                 				#Break if q is pressed
           break 											#
    else: 												#
        break 											#
cap.release()                                                                                                           					#Close the video file
cv2.destroyAllWindows()                                                                                                 					#All windows are destroyed if program is terminated
#################################################################################################
