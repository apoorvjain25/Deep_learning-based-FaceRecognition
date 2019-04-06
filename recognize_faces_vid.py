# python recognize_faces_video_file.py --encodings encodings.pickle --input videos/lunch_scene.mp4



#python C:\Python27\Test\deep_learning\face-recognition-opencv\recognize_faces_vid.py --encodings Test\deep_learning\face-recognition-opencv\encodings.pickle --input Advait\lunch_scene.mp4 -d 'hog'

import face_recognition

import argparse

import imutils

import pickle

import time

import cv2
import datetime


from time import gmtime, strftime
showdate = strftime("%Y-%m-%d", gmtime())
showtime = strftime("%H:%M:%S", gmtime())



#

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
 help="path to serialized db of facial encodings")

ap.add_argument("-i", "--input", required=True,
	help="path to input video")

ap.add_argument("-o", "--output", type=str,
 help="path to output video")

ap.add_argument("-y", "--display", type=int, default=1,
	help="whether or not to display output frame to screen")

ap.add_argument("-d", "--detection-method", type=str, default="cnn",
 help="face detection model to use: either `hog` or `cnn`")

args = vars(ap.parse_args())



# load the known faces and embeddings

print("[INFO] loading encodings...")

data = pickle.loads(open(args["encodings"], "rb").read())



print("[INFO] processing video...")

stream = cv2.VideoCapture(args["input"])

writer = None




while True:

	
	(grabbed, frame) = stream.read()



	if not grabbed:

		break


	# convert the input frame from BGR to RGB then resize it to have


	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	rgb = imutils.resize(frame, width=750)

	r = frame.shape[1] / float(rgb.shape[1])




	boxes = face_recognition.face_locations(rgb,
 model=args["detection_method"])

	encodings = face_recognition.face_encodings(rgb, boxes)

	names = []



	# loop over the facial embeddings

	for encoding in encodings:

		matches = face_recognition.compare_faces(data["encodings"],
 encoding)

		name = "Unknown"



		# check to see if we have found a match

		if True in matches:


			matchedIdxs = [i for (i, b) in enumerate(matches) if b]

			counts = {}



			for i in matchedIdxs:

				name = data["names"][i]

				counts[name] = counts.get(name, 0) + 1


			name = max(counts, key=counts.get)

		names.append(name)



	# loop over the recognized faces

	for ((top, right, bottom, left), name) in zip(boxes, names):

		# rescale the face coordinates

		top = int(top * r)

		right = int(right * r)

		bottom = int(bottom * r)

		left = int(left * r)


		
		# draw the predicted face name on the image

		cv2.rectangle(frame, (left, top), (right, bottom),
 (0, 255, 0), 2)

		y = top - 15 if top - 15 > 15 else top + 15

		x = bottom - 30
		z = bottom - 45
		#bottom - 30 if bottom - 30 > 30 else
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
 0.75, (0, 255, 0), 2)


		cv2.putText(frame, showtime, (bottom, x), cv2.FONT_HERSHEY_SIMPLEX,
 0.75, (0, 255, 0), 2)
		cv2.putText(frame, showdate, (z, bottom), cv2.FONT_HERSHEY_SIMPLEX,
 0.75, (0, 255, 0), 2)
	
	if writer is None and args["output"] is not None:

		fourcc = cv2.VideoWriter_fourcc(*"MJPG")

		writer = cv2.VideoWriter(args["output"], fourcc, 24,
 (frame.shape[1], frame.shape[0]), True)



	if writer is not None:

		writer.write(frame)



	if args["display"] > 0:

		cv2.imshow("Frame", frame)

		key = cv2.waitKey(1) & 0xFF



		# if the `q` key was pressed, break from the loop

		if key == ord("q"):

			break



# close the video file pointers

stream.release()


# check to see if the video writer point needs to be released

if writer is not None:

	writer.release()

