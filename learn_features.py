'''
Use this script to learn facial features of persons after 
having images of all persons stored in `photos` directory.

All the embeddings and features will be stored in the data.py
file that is loaded in direct_use.py
'''

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image

# Initializing MTCNN and InceptionResnetv1
mtcnn0 = MTCNN(image_size=240, margin=0, keep_all=False, min_face_size=40) # keep_all = False means if one img contains many faces then it will keep only 1 face of those
resnet = InceptionResnetV1(pretrained='vggface2').eval() # creating an instance of pre-trained InceptionResnetV1 model trained on the VGGFace2 dataset to extract embeddings

# Read data from directory
dataset = datasets.ImageFolder('photos') # photos directory path
idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} # new dictionary of indecis and class names

def collate_fn(x):
    return x[0]

loader = DataLoader(dataset, collate_fn=collate_fn)

name_list = [] # list of names corresponding to cropped photos
embedding_list = [] # list of embedding matrix after conversion from cropped faces to embedding matrix using resnet

for img, idx in loader: # img is in PIL format, here we are looping thru the images in the photos directory (loader)
    face, prob = mtcnn0(img, return_prob=True) # img passed into mtcnn0, face is the cropped face (note: in mtcnn0 it will return only ONE face)
    if face is not None and prob>0.92: # if face is not None means that there's a face that exists 
        emb = resnet(face.unsqueeze(0)) # pass the face into resnet, unsqueeze b/c resnet expects 4 dimensions (dimension of batch included). Result of this is an embedding
        embedding_list.append(emb.detach()) # .detach() to make requires_grad false
        name_list.append(idx_to_class[idx])

# save data
data = [embedding_list, name_list] # make a new list of the previous 2 lists
torch.save(data, 'data.pt') # saving to make this code reusable without having to go thru the previous loops and steps every single time which is very costly
print("[INFO] Features and embeddings of all faces learned and stored in data.pt file")