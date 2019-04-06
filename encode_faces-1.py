# python encode_faces.py --dataset dataset --encodings encodings.pickle

#please add dataset folder path above, which ever you want to use 
#in the form of different folders for each person with folder name as person name. 

from imutils import paths

import face_recognition

import argparse

import pickle

import cv2

import os


import numpy as np
import pandas as pd 


ap = argparse.ArgumentParser()

ap.add_argument("-i", "--dataset", required=True,
 help="path to input directory of faces + images")

ap.add_argument("-e", "--encodings", required=True,
 help="path to serialized db of facial encodings")

ap.add_argument("-d", "--detection-method", type=str, default="cnn",
 help="face detection model to use: either `hog` or `cnn`")

args = vars(ap.parse_args())

  


print("[INFO] quantifying faces...")

imagePaths = list(paths.list_images(args["dataset"]))




knownEncodings = []

knownNames = []



for (i, imagePath) in enumerate(imagePaths):
	
	# extract the person name from the image path
	
	print("[INFO] processing image {}/{}".format(i + 1,
 len(imagePaths)))
	
	name = imagePath.split(os.path.sep)[-2]

	

	
	image = cv2.imread(imagePath)
	
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	
	# detect the (x, y)-coordinates of the bounding boxes
	
 	# corresponding to each face in the input image
	
	boxes = face_recognition.face_locations(rgb,
model=args["detection_method"])

	
	# compute the facial embedding for the face
	
	encodings = face_recognition.face_encodings(rgb, boxes)

	
	
	for encoding in encodings:
				
		knownEncodings.append(encoding)
		
		knownNames.append(name)





print("[INFO] serializing encodings...")

data = {"encodings": knownEncodings, "names": knownNames}


df = pd.DataFrame(knownEncodings)
df.to_csv("C:\\Python27\\feature.csv")

df1 = pd.DataFrame(knownNames)
df1.to_csv("C:\\Python27\\Name.csv")

f = open(args["encodings"], "wb")

f.write(pickle.dumps(data))

f.close()