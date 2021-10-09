
# Team Gegabyte Final Code

# To run this code it is must to have some libraries beforehand.( those are - 'cv2','numpy','face_recogniition','os','dlib')


import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


path = 'images'                                                   # A path is created to "images" directory
photos = []
names = []
myList = os.listdir(path)                                         # "mylist" is the list of files in given path

for image_now in myList:                                          # image_now is a variable defined to get to all images in directory one by one.
    current_Img = cv2.imread(f'{path}/{image_now}')
    photos.append(current_Img)                                    #'.append' function adds new elements to the list
    photos.append(os.path.splitext(image_now)[0])


 ##  EncodeFaces is a function defined to find the encodings of photos available in 'images' directory


def encodeFaces(images):
    encode_1 = []                                                # "encode-1" is the blank list created to hold the encodings of each image.
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)                # 'cv2.cvtcolor' is a function used to convert image from one color space to another.Here cv2.BGR2RGB converts BGR image into RGB
        encode = face_recognition.face_encodings(img)[0]         #'face-encodings' is a function availabe in 'face_recognition' library.It encodes every image available in given directory
        encode_1.append(encode)
    return encode_1                                              # the function returns the final list of encodings of each faces( each face has a total of 128 encodings found through HOG algorithm avaiable in face-recognition library )


## 'CheckAttendance' is a function defined to store attendance log of employees

def CheckAttendance(name):
    with open('attendancelog.csv', 'r+') as p:
        list_1 = p.readlines()
        nameList = []
        #time_now = datetime.now()
        #date = time_now.strftime('%d/%m/%Y')
        for line in list_1:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time_now_1 = datetime.now()
            time = time_now_1.strftime('%H:%M:%S')
            date = time_now_1.strftime('%d/%m/%Y')
            p.writelines(f'\n{name},{time},{date}')


KnownList = encodeFaces(photos)

print('All photos are encoded !')

vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read()
    final_face = cv2.resize(frame, (0, 0), None, 0.4,0.4)
    final_face = cv2.cvtColor(final_face, cv2.COLOR_BGR2RGB)

    faceLocation = face_recognition.face_locations(final_face)
    encoding_face = face_recognition.face_encodings(final_face, faceLocation)

    for encodeFace, faceLoc in zip(encoding_face, faceLocation):
        found_match = face_recognition.compare_faces(KnownList, encodeFace)
        faceDis = face_recognition.face_distance(KnownList, encodeFace)
        index = np.argmin(faceDis)

        if found_match[index]:
            name = names[index]
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 2.5, x2 * 2.5, y2 * 2.5, x1 * 2.5
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            CheckAttendance(name)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) == 13:
        break

vid.release()
cv2.destroyAllWindows()