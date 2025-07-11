import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'ImagesAttendance'
images = [] #create list of images we are importing
classNames = [] #create a list of image names
myList = os.listdir(path) #makes a list of all the names in the ImagesAttendance folder
#print(myList)

for cl in myList:  #for each image in the folder
    curImg = cv2.imread(f'{path}/{cl}') #reads the current image
    images.append(curImg) #append our current image
    classNames.append(os.path.splitext(cl)[0]) #append the class name, also removes the .jpg at the end

print(classNames)

def findEncodings(images): #computes all the encodings of the images
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0] #finds the encoding of image
        encodeList.append(encode) #appends it to our list
    return encodeList #returns the encode list

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0]) #appends only the names in the list
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


encodeListKnown = findEncodings(images) #the amount of encodings made
print('Encoding Completed')

cap = cv2.VideoCapture(0) #initalizes webcam

while True:
    success, img = cap.read() #this will give us our image
    imgS = cv2.resize(img, (0, 0), None, 0.25,0.25) #reduces the size of the image to help process in real-time, 1/4 of the size
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB) #converts it to RGB

    facesCurrentFrame = face_recognition.face_locations(imgS) #finds faces in webcam
    encodesCurrentFrame = face_recognition.face_encodings(imgS, facesCurrentFrame) #encodes faces in webcam

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)


    cv2.imshow('Webcam', img)
    cv2.waitKey(1)