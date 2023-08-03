# AI-based Attendance System using Gaceial Recognition
This project aims to develop a system that stores embeddings (vector of features) of faces in a database and based on it, identify faces from a video stream in real-time and assign an ID to the face if it matches the embeddings of one of the faces in the database. All the information of when an employee enters the company, the time employee leaves, and the time he/she spent in the company are automated and stored in an excel file.

[Watch video a demonstration.](https://www.youtube.com/watch?v=MxFXoTrUpw0)

Face detection is carried out by **MTCNN** . Face recognition is carried out using **Facenet**. Both of which are open source and are available through _facenet_pytorch_ github repo.

Features include:
* High accuracy for face detection due to state of the art algorithms used
* Marking the exact time an employee arrives at a company/institution on a given day
* Marking the exact time an employee leaves the company/institution on a given day
* Calculating how much time an employee spent in the company/instituion on a given day
* Easily add new faces of employees in at any moment while the application is running
* Ability not to be fooled by simply showing an image of an employee to the camera
* Update an excel file in real-time with latest employee attendance information
* Ability to recognise multiple faces simultaneously
* Visual feedback when an employee is marked as arrived, or departed
* Export a unique excel file every day with the date in the file name on each new day (feature added after the video demonstration. Therefore, not present in the video)

Feel free to download the packages in the **requirements.txt** file to run the program locally on your computer.
Note: To avoid errors while running the code if you are using Windows or Linux, at the very end of the code, replace:
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
