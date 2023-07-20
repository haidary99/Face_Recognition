#!/usr/bin/env python
# coding: utf-8

# https://www.youtube.com/watch?v=IT53xBR1A7M&list=PLqUHmcsDDjLjFSo4iumsXGhQoaNu-aXUF

# ### This jupyter notebook is used to detect faces in images (using MTCNN) stored in `photos` directory and extract their features (using InceptionResnetV1) and store them in `data.pt` file. This file is then loaded later in order to extract the features of faces stored in it and compare it with the face(s) detected in each live frame from a camera stream. If there is a match according to some threshold, then it will label the person in the frame.
# ***

# In[10]:


from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import time
import os


# In[11]:


# Initializing MTCNN and InceptionResnetv1

mtcnn0 = MTCNN(image_size=240, margin=0, keep_all=False, min_face_size=40) # keep_all = False means if one img contains many faces then it will keep only 1 face of those
mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=150) # keep_all = True keeps all faces in a form of list, Also changed mmin_face_size to 150 to ignore small faces in frame which might be difficult to classify
resnet = InceptionResnetV1(pretrained='vggface2').eval() # creating an instance of pre-trained InceptionResnetV1 model trained on the VGGFace2 dataset to extract embeddings


# In[12]:


# Read data from directory

dataset = datasets.ImageFolder('photos') # photos directory path
idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} # new dictionary of indecis and class names
idx_to_class


# In[13]:


def collate_fn(x):
    return x[0]

loader = DataLoader(dataset, collate_fn=collate_fn)
loader


# In[14]:


name_list = [] # list of names corresponding to cropped photos
embedding_list = [] # list of embedding matrix after conversion from cropped faces to embedding matrix using resnet


# In[15]:


for img, idx in loader: # img is in PIL format, here we are looping thru the images in the photos directory (loader)
    face, prob = mtcnn0(img, return_prob=True) # img passed into mtcnn0, face is the cropped face (note: in mtcnn0 it will return only ONE face)
    if face is not None and prob>0.92: # if face is not None means that there's a face that exists 
        emb = resnet(face.unsqueeze(0)) # pass the face into resnet, unsqueeze b/c resnet expects 4 dimensions (dimension of batch included). Result of this is an embedding
        embedding_list.append(emb.detach()) # .detach() to make requires_grad false
        name_list.append(idx_to_class[idx])

# save data
data = [embedding_list, name_list] # make a new list of the previous 2 lists
torch.save(data, 'data.pt') # saving to make this code reusable without having to go thru the previous loops and steps every single time which is very costly


# # Using webcam to recognize the face

# In[20]:


# Loading data.pt file
load_data = torch.load('data.pt')
embedding_list = load_data[0] 
name_list = load_data[1]

cam = cv2.VideoCapture(0) # initializing the camera from CV, 0 means the default webcam in the device

while True: # read all the frames in the video
    ret, frame = cam.read() # return is true or false, if the webcam succesfully captures an image then it is true otherwise false. frame reperesnts a single frame (image) captured from the webcam using VideoCapture object
    if not ret:
        print("Fail to grab frame, try again")
        break
    
    img = Image.fromarray(frame) # Image class is part of PIL, .fromarray() is a method that creates an "Image" object from a numpy array 
    img_cropped_list, prob_list = mtcnn(img, return_prob=True) # pass the image into mtcnn, mtcnn will return multiple faces if the image contain multiple faces, also return all the probabilities for all the faces

    if img_cropped_list is not None: # if the image has at least one face
        boxes, _ = mtcnn.detect(img) # return the boxes of faces 
        # print(boxes[0][0])

        for i, prob in enumerate(prob_list): # loop thru the prob list
            if prob > 0.90: 
                emb = resnet(img_cropped_list[i].unsqueeze(0)).detach()
                
                dist_list = [] # for the distance between the current embedding and embeddings of faces in the photos directory (similarity). Minimum distance is used to identify the person

                for idx, emd_db in enumerate(embedding_list):
                    dist = torch.dist(emb, emd_db).item() # calc. distance betweem current embedding and embeddings stored embedding_list
                    dist_list.append(dist)

                # print(f"Minimum distance array:\n{dist_list}")
                min_dist = min(dist_list) # get minimum dist value
                formatted_min_dist = f'{min_dist:.4f}' # used for printing on the frame
                min_dist_idx = dist_list.index(min_dist) # get minimum dist index (where the min dist is located in the array)
                name = name_list[min_dist_idx] # get name corresponding to minium dist ##########
                name = name.replace('_', ' ')

                box = boxes[i]

                original_frame = frame.copy() # storing a copy of frame before drawing on it
        
                if min_dist<0.90:
                    frame = cv2.putText(frame, name+' '+str(formatted_min_dist), (int(box[0])-5, int(box[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

                frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255,0,0), 3)

    cv2.imshow("IMG", frame)

    k = cv2.waitKey(1)
    # Press q or esc to quit, space bar to add & save new image
    if k == ord('q'):
        break # Quit the program

    elif k == 27: # 27 is for esc key
        break

    elif k == 32: # 32 is the ASCII of space bar
        print('Enter your name: ')
        name = input()
        if name == "": ###### I ADDED THESE 2 LINES IN CASE INPUT WAS EMPTY IT DOESN'T SAVE TEH IMAGE
            continue

        # Create directory if not exists
        if not os.path.exists('photos/'+name):
            os.mkdir('photos/'+name)

        img_name = f"photos/{name}/{int(time.time())}.jpg"
        cv2.imwrite(img_name, original_frame)
        print(f"saved: {img_name}")

cam.release()

for i in range(1):
    cv2.destroyAllWindows()
    cv2.waitKey(1)

# OR the below method, but below method requires one more key press to destroy the window while the above loop automatically closes the window without extra key press

# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.waitKey(1) # This is the only extra line u need on top of windows os uses to work properly on mac