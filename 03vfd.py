from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
import numpy as np
import cv2
from PIL import Image

mtcnn = MTCNN(image_size=240,keep_all=True,margin=0, min_face_size=20)
resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion

load_data=torch.load('data.pt')
embedding_list=load_data[0]
name_list=load_data[1]
count=0 
cam=cv2.VideoCapture("videos/vid.mp4") 

while(True):

    ret, frame = cam.read()
    if not ret: 
        print("fail to grab frame")
        break
    img=Image.fromarray(frame)
    img_cropped_list,prob_list=mtcnn(img,return_prob=True)
    # print(img_cropped.numpy())
    # cv2.imshow("ci",img_cropped.numpy().transpose(1,2,0))
    if img_cropped_list is not None:
        boxes, probs, landmarks =mtcnn.detect(frame, landmarks=True)
        for i,prob in enumerate(prob_list):
            if prob>.90:
                emb=resnet(img_cropped_list[i].unsqueeze(0)).detach()
                dist_list = [] # list of matched distances, minimum distance is used to identify the person
                
                for idx, emb_db in enumerate(embedding_list):
                    dist = torch.dist(emb, emb_db).item()
                    dist_list.append(dist)
                    
                    
                min_dist = min(dist_list) 
                min_dist_idx =dist_list.index(min(dist_list)) 
                name=name_list[min_dist_idx]
                # print(name)
                cbox=boxes[i]
                original_frame=frame.copy()
                
                
                try:
                    for box, prob, ld in zip(boxes, probs, landmarks):
                        # Draw rectangle on frame
                        
                        cv2.rectangle(frame,
                                    (box[0], box[1]),
                                    (box[2], box[3]),
                                    (0, 0, 255),
                                    thickness=2)
                            # if min_dist<0.90:
                            #     cv2.putText(frame,name+''+str(min_dist), (box[2], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        # print(sum(cbox),sum(box))
                        if sum(cbox)==sum(box) and min_dist<0.90:
                            cv2.putText(frame, str(
                                name), (box[2], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                except :
                    pass   
                
        cv2.imshow("img",frame)
    
        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 30: # Take 30 face sample and stop video
            break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
