import cv2
import datetime
import imutils
import numpy as np
import pandas as pd
import pygame
from collections import defaultdict
from nms import non_max_suppression_fast
from centroidtracker import CentroidTracker
protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

CLASSES = ["background", "aeroplane", "bicycle", "bird","boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

tracker = CentroidTracker(maxDisappeared=10, maxDistance=90)

def main():
    cap = cv2.VideoCapture('test_video.mp4')
    # cap=cv2.VideoCapture(0)
    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0
    lock=0
    tmp=[]
    elapsed_dict=defaultdict(list)
    object_id_list = []
    dtime = dict()
    dwell_time = dict()
    my_dict = {"Id":[],"Time":[],"Elapsed_time":[]}
    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=600)
        total_frames = total_frames + 1

        (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

        detector.setInput(blob)
        person_detections = detector.forward()
        rects = []
        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(person_detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")
                rects.append(person_box)

        boundingboxes = np.array(rects)
        boundingboxes = boundingboxes.astype(int)
        rects = non_max_suppression_fast(boundingboxes, 0.3)

        objects = tracker.update(rects)
        for (objectId, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2) 
            y2 = int(y2)

            if objectId not in object_id_list:
                object_id_list.append(objectId)
                now=datetime.datetime.now()
                dtime[objectId]=now
                dwell_time[objectId] = 0
                lock=0
                tmp.append(0)
                time = now.strftime("%y-%m-%d %H:%M:%S")
                my_dict["Id"].append((str(objectId)))
                my_dict["Time"].append(str(time))
            else:
                curr_time = datetime.datetime.now()
                old_time = dtime[objectId]
                time_diff = curr_time - old_time
                dtime[objectId] = datetime.datetime.now()
                sec = time_diff.total_seconds()
                dwell_time[objectId] += sec
                #print(dwell_time[objectId])
                if(int(dwell_time[objectId])>50 and lock==0):
                    print('Anomaly detected')
                    lock=1
                    pygame.init()
                    pygame.mixer.init()
                    alarm_sound = pygame.mixer.Sound('alarm.mp3')  
                    alarm_sound.play() 
                elapsed_dict[objectId].append(int(dwell_time[objectId])) 
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = "{}|{}".format(objectId, int(dwell_time[objectId]))
            cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            if lock == 1:
                anomaly_text = "Anomaly Detected!"
                cv2.putText(frame, anomaly_text, (x1 + 50, y1 - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0, 0, 255), 2)
        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (total_frames / time_diff.seconds)

        fps_text = "FPS: {:.2f}".format(fps)

        cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        cv2.imshow("Application", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            mydict=dict(elapsed_dict)
            #print((mydict[0][-1]))
            tmp_list=[mydict[x][-1] for x in range(len(mydict))]
            print(tmp_list)
            my_dict={"Id":my_dict["Id"],"Time":my_dict["Time"],"Elapsed_time":tmp_list}
            print(my_dict)
            df=pd.DataFrame.from_dict(my_dict)    
            df.to_csv('dwell_time_calculation.csv', index=False)   
            break

    cv2.destroyAllWindows()


main()
