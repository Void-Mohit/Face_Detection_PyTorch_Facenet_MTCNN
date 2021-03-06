import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
import os

class FaceDetector(object):
    """
    Face detector class
    """

    def __init__(self, mtcnn):
        self.mtcnn = mtcnn

    def run(self):
        """
            Run the FaceDetector and draw landmarks and boxes around detected faces
        """
        cap = cv2.VideoCapture("videos/akkai.mp4")
        face_id = input('\n enter user id end press <return> ==>  ')

        print("\n [INFO] Initializing face capture. Look the camera and wait ...")
        dname=""+str(face_id)
        path = os.path.join("photos/", dname)
        try:  
            os.mkdir(path)  
        except OSError as error:  
            print(error)    
        # Initialize individual sampling face count
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret: 
                print("fail to grab frame")
                break
            try:
                # detect face box, probability and landmarks
                boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)
                # draw on frame
                # self._draw(frame, boxes, probs, landmarks)
                if probs>.98:
                    for box, prob, ld in zip(boxes, probs, landmarks):
                        # Draw rectangle on frame
                        cv2.rectangle(frame,
                                    (box[0], box[1]),
                                    (box[2], box[3]),
                                    (0, 0, 255),
                                    thickness=2)
                        count += 1

                        # print(int(box[0]),int(box[1]),int(box[2]), int(box[3]))
                        frame=frame[int(box[1]):int(box[3]),int(box[0]): int(box[2])]
                        # frame=frame[106:254,311:424]
                        
                        cv2.imwrite("photos/"+dname+"/frame"+ str(count) + ".jpg",frame)
                        # name = 'dataset/'+ str(face_id) +"/tempframe" +str(count) + '.jpg'
                        # print ('Creating...' + name)  
                        # cv2.imwrite(name,frame) 
                        

            except:
                pass
            
            
            k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
            if k == 27:
                break
            elif count >= 30: # Take 30 face sample and stop video
                break

            # Show the frame
            cv2.imshow('Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        
        
# Run the app
mtcnn = MTCNN()
fcd = FaceDetector(mtcnn)
fcd.run()