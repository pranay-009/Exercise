


import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import time
from PIL import ImageGrab


mp_drawing=mp.solutions.drawing_utils
mp_pose=mp.solutions.pose



def angle(a,b,c):
    a=np.array(a)
    b=np.array(b)
    c=np.array(c)
    
    arc=np.arctan2(c[1]-b[1],c[0]-b[0])-np.arctan2(a[1]-b[1],a[0]-b[0])
    angl=np.abs(arc*180.0/np.pi)
    if angl>180:
        angl=360-angl
    return angl    
        


cap=cv2.VideoCapture(0) #to get the video capture object we are since we have one camera source i select my web cam so 0
count=0 #this is a flag variable which basically does the job of re-initiating the timer when c is 0 we can start the timer and change the flag to 0 so that utill the end we dont update the start time
flag=0#flag varable
end=0# it store the end time variable 
start=0#stores the start time 
rep=0# store number of reps
rep_time=8.0# min time for staying bend
out = cv2.VideoWriter(r'E:/recorded_video/video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 5, (580,420))
with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
    count=0
    while True:
        #fetch video frames from front camera
        frame=ImageGrab.grab(bbox =(670,350,1250, 770))
        frame=np.array(frame)
        #ret,frame=cap.read()
        image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable=False
        #store the results of detected features in result
        result=pose.process(image)
        image.flags.writeable=True
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        #draw the landmarks for pose detection
        mp_drawing.draw_landmarks(image,result.pose_landmarks,mp_pose.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2))
        try:
            #store the coordinates of left leg(hip,knee,foot)
            a=list([result.pose_landmarks.landmark[23].x,result.pose_landmarks.landmark[23].y])
            b=list([result.pose_landmarks.landmark[25].x,result.pose_landmarks.landmark[25].y])
            c=list([result.pose_landmarks.landmark[27].x,result.pose_landmarks.landmark[27].y])
            #store the coordinates of right leg(hip,knee,foot)
            p=list([result.pose_landmarks.landmark[24].x,result.pose_landmarks.landmark[24].y])
            q=list([result.pose_landmarks.landmark[26].x,result.pose_landmarks.landmark[26].y])
            r=list([result.pose_landmarks.landmark[28].x,result.pose_landmarks.landmark[28].y])
            #calculate the angle between the knee and the thigh and calves :we do this to measure the bend
            right=angle(a,b,c)
            left=angle(p,q,r)
            #put the measured angle in the knee co-ordinates
            #cv2.putText(image,str(round(right,0)),tuple(np.multiply(b, [640, 480]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
            #cv2.putText(image,str(round(left,0)),tuple(np.multiply(q, [640, 480]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
            # now if the bend angle is less than 140 then enter the loop
            if right<140 and left<140:
                # we introduce the condition count ==0 to check if its the first time frame detects the bend so we can store the start value of timer
                if count==0:
                    start=time.time()
                    #we update count to 1 so that i does not update the start time unless the rep ends
                    count=1
                
                cv2.putText(image,str(round(right,0)),tuple(np.multiply(b, [640, 480]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image,str(round(left,0)),tuple(np.multiply(q, [640, 480]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                #counts how long you are able to stay bend
                rp_cnt=round(time.time()-start,0)
                cv2.putText(image,"Timecount:"+str(rp_cnt),(400,25),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                #since there can be number of frames in one second we want to set a flag varible which updates once after completion of a rep
                if(rp_cnt==rep_time and flag==0):
                    flag=1
                    #update rep by 1
                    rep=rep+1
            #when angle between thigh and claves is greater than 140
            elif right>=140 or left>=140:
                cv2.putText(image,str(round(right,0)),tuple(np.multiply(b, [640, 480]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(image,str(round(left,0)),tuple(np.multiply(q, [640, 480]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
                
                #store the end time
                end=time.time()
                
                if (end-start)<rep_time and start!=0:
                    cv2.rectangle(image, (0,0),(380,70),(0,255,0), thickness=-1)
                    cv2.putText(image,"KEEP YOUR KNEE BEND!",(65,20),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    #since the person failed to stay bend we restart the process so we again update the flag to 0
                    count=0
                    
                elif (end-start)>=rep_time:
                    #now the rep is complete so we want to update falg variable for next rep
                    count=0
                    flag=0
                    
        except: 
            pass
        cv2.putText(image,"rep count:"+str(rep),(65,40),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        out.write(image)
        cv2.imshow("front cam",image) 
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()




