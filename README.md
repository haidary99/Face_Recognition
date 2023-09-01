# AI-based Attendance System using Facial Recognition
This project aims to develop a system that stores embeddings (vector of features) of employee faces which are learned by utilizing AI algorithms in a database and based on this information, identify faces from a video stream in real-time and assign the appropriate ID to the face if it matches the embeddings of one of the faces in the database. All the information of when an employee enters the company, the time employee leaves, and the time he/she spent in the company are automated and stored in a csv file.

[Watch video a demonstration.](https://youtu.be/NxIFCbIStIQ)

Face detection is carried out by **MTCNN** . Face recognition is carried out using **Facenet**. Both of which are open source and are available through _facenet_pytorch_ github repo.

Features include:
* High accuracy for face detection due to state of the art algorithms used
* Marking the exact time an employee arrives at a company/institution on a given day
* Marking the exact time an employee leaves the company/institution on a given day
* Calculating how much time an employee spent in the company/instituion on a given day
* Easily add new faces of employees in at any moment while the application is running
* Ability not to be fooled by simply showing an image of an employee to the camera
* Update an csv file in real-time with latest employee attendance information
* Ability to recognise multiple faces simultaneously
* Visual feedback when an employee is marked as arrived, or departed
* Export a unique csv file every day with the date in the file name on each new day

Feel free to download the packages in the **requirements.txt** file to run the program locally on your computer.
<br>
You can either use the Jupyter notebook ``face_reco.ipynb`` which contains all the required code to learn the facial features in the ```photos``` directory and start the program. <br>
Or alternatively, use the two Python scripts to first learn the features ``(learn_features.py)``, and then directly start the program without having to learn the features every single time ``(direct_use.py)``.
<br>

**Note:** To avoid errors while running the code if you are using Windows or Linux, at the very end of the code in the ``face_reco.ipynb`` or ``direct_use.py`` files, replace:
```
for i in range(1):
    cv2.destroyAllWindows()
    cv2.waitKey(1)
```

with:
```
cv2.waitKey(0)
cv2.destroyAllWindows()
```
And change ``cam = cv2.VideoCapture(2)`` to ``cam = cv2.VideoCapture(0)``
