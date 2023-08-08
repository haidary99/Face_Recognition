"""
This file is used to directly utilize `InceptionResnetV1` in order to compare face(s) in 
frame with saved embeddings in `data.pt`.
This file does not train on the available dataset of faces! You need to use `learn_featurs.py`
for that purpose.
"""


from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import cv2
import time
import os
from datetime import timedelta, datetime, date
import pandas as pd

mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=300) # keep_all = True keeps all faces in a form of list, Also changed mmin_face_size to 150 to ignore small faces in frame which might be difficult to classify
resnet = InceptionResnetV1(pretrained='vggface2').eval() # creating an instance of pre-trained InceptionResnetV1 model trained on the VGGFace2 dataset to extract embeddings

import pandas as pd
import time
from datetime import timedelta, datetime, date

def update_status(row):
    '''
    Change status from absent to present or departed based on other column values
    '''
    if pd.notna(row['arrival_date']):
        if pd.notna(row['departure_date']):
            # print("changed status to departed")
            return 'departed'
        # print("changed status to present")
        return 'present'
    else:
        # print("No change in status")
        return row['status']


def update_time_diff(row):
    '''
    calculate time_difference column based on other column values
    '''
    if pd.notna(row['arrival_date']):
        if pd.notna(row['departure_date']):
            time_format = "%H:%M:%S"
            arrival_datetime = datetime.strptime(row['arrival_time'], time_format)
            departure_datetime = datetime.strptime(row['departure_time'], time_format)
            time_diff = (departure_datetime - arrival_datetime).seconds
            time_diff_seconds = timedelta(seconds=time_diff)
            return time_diff_seconds
    else:
        # print("No change in status")
        return row['time_difference']


# in case program stops working on a specific day and is reloaded, the old info should not be overwritten! therefore we define the below function:
# Function to save attendance data to an Excel file with the current date in the file name
def save_attendance_to_excel(df_attendance):
    today = date.today().strftime('%Y-%m-%d')
    excel_file = f'attendance_{today}.csv'

    if os.path.exists(excel_file):
        # If the file already exists, read its content into a DataFrame
        existing_df = pd.read_csv(excel_file)
        # print(f"\nExisting file:\n{existing_df}\n")
        # Merge the existing DataFrame with the new attendance data based on the "Name" column
        updated_df = existing_df.merge(df_attendance, on="Name", how="left", suffixes=('', '_new'))

        # Replace the existing columns with the new columns where there are no missing values
        for col in df_attendance.columns:
            if f"{col}_new" in updated_df.columns:
                updated_df[col] = updated_df[col].fillna(updated_df[f"{col}_new"])

        # Drop the temporary columns with "_new" suffix
        updated_df = updated_df.drop([col for col in updated_df.columns if col.endswith('_new')], axis=1)

        # Change status from absent to present or departed based on other column values
        updated_df["status"] = updated_df.apply(update_status, axis=1)

        # Calculate time_difference column based on other column values
        updated_df["time_difference"] = updated_df.apply(update_time_diff, axis=1)

        # Save the updated DataFrame back to the Excel file, overwriting its content
        updated_df.to_csv(excel_file, index=False)

        # print(f"\ndf_attendnace:\n {df_attendance}\n")
        # print(f"\nupdated_df:\n {updated_df}\n")

    else:
        # If the file does not exist, save the attendance data to a new Excel file
        df_attendance.to_csv(excel_file, index=False)


# Loading data.pt file
load_data = torch.load('data.pt')
embedding_list = load_data[0] 
name_list = load_data[1]

cam = cv2.VideoCapture(2)  # initializing the camera from CV, 0 means the default webcam in the device

# this next line is for Iriun application to use iphone camera
# cam = cv2.VideoCapture(0) 

# These 3 lines are for droidcam app to use iphone camera
# cam = cv2.VideoCapture(1)
# address = "http://192.168.1.105:4747/video"
# cam.open(address)

names = [name.replace('_', ' ') for name in name_list]
attendance_dict = {name: {"status": "absent", "arrival_date": None, "arrival_time": None, "departure_date": None, "departure_time": None, "time_difference": None} for name in names} # Initialize attendance dictionary which has all the names in the db as keys and another dictionary as value

# Initialize counter and maximum consecutive frames
consecutive_frames_dict = {name: 0 for name in names} # all names in db as keys and 0 for their values
max_consecutive_frames = 50 # increase/decrease this to modify time needed to take attendance

# Convert the attendance_dict to a Pandas DataFrame
df_attendance = pd.DataFrame.from_dict(attendance_dict, orient='index')

# Reset the index to get the names as a column instead of an index
df_attendance.reset_index(inplace=True)
df_attendance.rename(columns={'index': 'Name'}, inplace=True)

# Print the DataFrame
# print("\nAttendance information before program starts:")
# print(df_attendance)

present_time_dict = {name: None for name in names} # used to calculate time diff. between arrival and departure

# bool_dict and timer_dict is used for changing status back to absent if employee did not depart (forgot to depart)
timer_dict = {name: None for name in names}
bool_dict = {name: False for name in names}

while True: # read all the frames in the video
    ret, frame = cam.read() # return is true or false, if the webcam succesfully captures an image then it is true otherwise false. frame reperesnts a single frame (image) captured from the webcam using VideoCapture object
    if not ret:
        print("\nFail to grab frame, try again")
        break
    
    img = Image.fromarray(frame) # Image class is part of PIL, .fromarray() is a method that creates an "Image" object from a numpy array 
    img_cropped_list, prob_list = mtcnn(img, return_prob=True) # pass the image into mtcnn, mtcnn will return multiple faces if the image contain multiple faces, also return all the probabilities for all the faces
    
    if img_cropped_list is None:
        # If there's no face, show message on frame re-initialize the counter back to 0. This is done to make sure that a face is detected in CONSECUTIVE frames if not, the counter is back to 0
        frame = cv2.putText(frame, "No face detected", (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        try:
            consecutive_frames_dict[name] = 0
        except NameError:
            # Handle the NameError here
            print("NameError occured")
            pass
    
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
                # formatted_min_dist = f'{min_dist:.4f}' # used for printing on the frame
                min_dist_idx = dist_list.index(min_dist) # get minimum dist index (where the min dist is located in the array)
                name = names[min_dist_idx] # get name corresponding to minium dist ##########
                name = name.replace('_', ' ')

                box = boxes[i]
 
                if name in consecutive_frames_dict: # when a face is detected, we retrieve the recognized name
                    consecutive_frames_dict[name] += 1
                    if consecutive_frames_dict[name] >= max_consecutive_frames:
                        # If the counter reaches or exceeds the max_consecutive_frames, we update the attendance status of the person in the attendance_dict from "absent" to "present"
                        if attendance_dict[name]["status"] == "absent":
                            
                            current_date = time.strftime("%Y-%m-%d")
                            current_time = time.strftime("%H:%M:%S")

                            print(f"\n[INFO] {name} arrived at: {current_time}")

                            # Update the date and specific time of face detection
                            attendance_dict[name]["arrival_time"] = current_time
                            attendance_dict[name]["arrival_date"] = current_date

                            # Update status
                            attendance_dict[name]["status"] = "present"

                            present_time_dict[name] = time.time() # used to calculate time diff. between arrival and departure
                            
                            # bool_dict and timer_dict is used for changing status back to absent if employee did not depart (forgot to depart)
                            bool_dict[name] = True
                            timer_dict[name] = time.time() # Record the time when the status was changed to "present"

                            # Update the consecutive_frames_dict[name] so that status is not changed immediately to departed 
                            consecutive_frames_dict[name] = 0

                            # Update the DataFrame with the latest attendance information
                            df_attendance["status"] = df_attendance["Name"].map(lambda name: attendance_dict[name]["status"])
                            df_attendance["arrival_date"] = df_attendance["Name"].map(lambda name: attendance_dict[name]["arrival_date"])
                            df_attendance["arrival_time"] = df_attendance["Name"].map(lambda name: attendance_dict[name]["arrival_time"])

                        elif consecutive_frames_dict[name] >= (max_consecutive_frames): ###### DONT FORGET TO REMOVE 1000
                            # If the current frame number minus the last_present_frame is greater than or equal to max_consecutive_frames,
                            # we change the status back to "departed" after a certain number of frames have passed since the employee's last presence.
                            if attendance_dict[name]["status"] == "present":

                                current_date = time.strftime("%Y-%m-%d")
                                current_time = time.strftime("%H:%M:%S")

                                print(f"\n[INFO] {name} departed at: {current_time}")

                                # Update the date and specific time of face detection
                                attendance_dict[name]["departure_time"] = current_time
                                attendance_dict[name]["departure_date"] = current_date

                                # Update status
                                attendance_dict[name]["status"] = "departed"

                                bool_dict[name] = False

                                # Update the consecutive_frames_dict[name]
                                consecutive_frames_dict[name] = 0

                                # Calculate time difference between arrived (present) and departed
                                arrival_time = attendance_dict[name]["arrival_time"]
                                departure_time = attendance_dict[name]["departure_time"]
                                time_format = "%H:%M:%S"
                                arrival_datetime = datetime.strptime(arrival_time, time_format)
                                departure_datetime = datetime.strptime(departure_time, time_format)
                                time_difference_seconds = (departure_datetime - arrival_datetime).seconds
                                print(f"time_difference_seconds: {time_difference_seconds}")

                                # Add the "time_difference" key to the attendance_dict
                                attendance_dict[name]["time_difference"] = timedelta(seconds=time_difference_seconds)
                                print(f"timedelta: {attendance_dict[name]['time_difference']}")

                                # Update the DataFrame with the latest attendance information
                                df_attendance["status"] = df_attendance["Name"].map(lambda name: attendance_dict[name]["status"])
                                df_attendance["departure_date"] = df_attendance["Name"].map(lambda name: attendance_dict[name]["departure_date"])
                                df_attendance["departure_time"] = df_attendance["Name"].map(lambda name: attendance_dict[name]["departure_time"])
                                df_attendance["time_difference"] = df_attendance["Name"].map(lambda name: attendance_dict[name]["time_difference"].__str__()[0:8])
                            
                    # Export the DataFrame to a CSV or xlsx (excel) file
                    save_attendance_to_excel(df_attendance)

                else:
                    consecutive_frames_dict[name] = 0

                original_frame = frame.copy() # storing a copy of frame before drawing on it
        
                if min_dist<0.90:
                    # frame = cv2.putText(frame, name+' '+str(formatted_min_dist), (int(box[0])-5, int(box[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                    # Display employee name and status on the frame
                    if attendance_dict[name]["status"] == "present":
                        status_color = (0, 255, 0)
                        status_text = "Arrived"
                        frame = cv2.putText(frame, f"{name} - {status_text}", (int(box[0])-5, int(box[1])-12), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2, cv2.LINE_AA)
                        frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), status_color, 3)

                    elif attendance_dict[name]["status"] == "departed":
                        status_color = (255, 0, 0)
                        status_text = "Departed"
                        frame = cv2.putText(frame, f"{name} - {status_text}", (int(box[0])-5, int(box[1])-12), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2, cv2.LINE_AA)
                        frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), status_color, 3)

                    else:
                        status_color = (0, 0, 255)
                        status_text = "Absent"
                        frame = cv2.putText(frame, f"{name} - {status_text}", (int(box[0])-5, int(box[1])-12), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2, cv2.LINE_AA)
                        frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), status_color, 3)

    cv2.imshow("IMG", frame)

    # If employee forgot to make departure (remained as arrived), mark him back as absent
    for name, Bool in bool_dict.items():
        if Bool:
            if (timer_dict[name] is not None) and (time.time() - timer_dict[name] >= ((24*60*60) - (5*60))): # 23 hours and 55 minutes
                # Change the status back to "absent" if 23 hours have passed and the status was not changed to "departed"
                if attendance_dict[name]["status"] == "present":
                    print(f"\n[INFO] {name} status automatically changed to 'Absent'")
                    attendance_dict[name]["status"] = "absent"
                    
                    # Update the DataFrame with the latest attendance information
                    df_attendance["status"] = df_attendance["Name"].map(lambda name: attendance_dict[name]["status"])
                    
                    # Reset the timer_dict entry for this employee
                    timer_dict[name] = None

                    # Export the DataFrame to a CSV or xlsx (excel) file
                    save_attendance_to_excel(df_attendance)

    k = cv2.waitKey(1)
    # Press q or esc to quit, space bar to add & save new image
    if k & 0xFF == ord('q'):
        break # Quit the program

    elif k & 0xFF == 27: # 27 is for esc key
        break

    elif k & 0xFF == 32: # 32 is the ASCII of space bar
        print('Enter your name: ')
        name = input()
        if name == "": ###### I ADDED THESE 2 LINES IN CASE INPUT WAS EMPTY IT DOESN'T SAVE TEH IMAGE
            continue

        # Create directory if class/person does not exist in photos directory
        if not os.path.exists('photos/'+name):
            os.mkdir('photos/'+name) 

        img_name = f"photos/{name}/{int(time.time())}.jpg"
        cv2.imwrite(img_name, original_frame)
        print(f"saved: {img_name}")

cam.release()

# Print the DataFrame
# print("\nAttendance information after program stopped:")
# print(df_attendance)

for i in range(1):
    cv2.destroyAllWindows()
    cv2.waitKey(1)

# OR the below method, but below method requires one moqre key press to destroy the window while the above loop automatically closes the window without extra key press

# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.waitKey(1) # This is the only extra line u need on top of windows os uses to work properly on mac