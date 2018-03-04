import cv2

#Cascade Intution Will be in GIT under the name (Module 1 Notes) incase you want to refer
#Cascade is not deep neural network but feature set

#we want to detect face and eye so we will load 2 cascades one for eye and face


face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')

#now code to detect and represent all the faces and eyes in the faces

def detect(gray,frame):
	#why 2 arguments : grey to detect and the frame to display boxes on the original
	#x->top left y->topleft h->height of box and w->width of the rectangle
	#many faces so we will put them all in tuple
	#use method detectMultiScale(param1,pram2,param3)
	#param1: image Param2: reduce the image by a factor of 1.3 param3 for a pixel to be accepted the number of neighbours to be accepted
	faces=face_cascade.detectMultiScale(gray, 1.3, 5)
	#now itirate through the faces detected (itirate the tuple)
	for (x,y,w,h) in faces:
		#rectangle(param1,param2,param3,param4,param5)
		#param1 : frame
		#param2: topleft corner
		#param3: bottom right corner
		#param4: color
		#param5 : thickness of the box
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		#detect eyes within in the face rectangle
		#roi==Region of intrest
		roi_gray=gray[y:y+h,x:x+w]
		roi_color=frame[y:y+h,x:x+w]
		eyes=eye_cascade.detectMultiScale(roi_gray,1.1,3)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	return frame

#stream Data! and use the detect 

video_capture=cv2.VideoCapture(0)
while True:
	#_ is actually 'ret' but we dont need it
	_,frame=video_capture.read()
	#convert color image into black and white
	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	canvas = detect(gray,frame)
	#imshow-->display the frames in one box
	cv2.imshow('Video',canvas)
	if cv2.waitKey(1) & 0xFF ==ord('q'):
		break

video_capture.release()
cv2.destroyAllWindows()










