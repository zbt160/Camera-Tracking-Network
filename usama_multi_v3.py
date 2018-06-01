#NOTE: Code works if person doesn't leave the FoV
import cv2
import imutils
import sys
from copy import deepcopy

cam1_list = []#Cam_1 id
cam2_list = []#Cam_2 id
central_list = []#Global id
global_count = 0

def main(cam1_path, cam2_path):
	#Read video object
	cam1_video = cv2.VideoCapture(cam1_path)
	cam2_video = cv2.VideoCapture(cam2_path)
	
	# Exit if video not opened.
	if not cam1_video.isOpened():
		print("Could not open cam1_video")
		sys.exit()
	if not cam2_video.isOpened():
		print("Could not open cam2_video")
		sys.exit()

	#Read first frame.
	ok, cam1_image = cam1_video.read()
	if not ok:
		print('Cannot read cam1_video file')
		sys.exit()
	
	ok, cam2_image = cam2_video.read()
	if not ok:
		print('Cannot read cam2_video file')
		sys.exit()
	
	#initialize camera 1, 2 trackers here
	cam1_tracker = cv2.MultiTracker_create()
	cam2_tracker = cv2.MultiTracker_create()
	
	cam1_personsTracked = []
	cam2_personsTracked = []
	#cam1_tracker.add(cam1_image, cam1_personsTracked);
	#cam2_tracker.add(cam2_image, cam2_personsTracked);
	
	#For each frame call individual trackers and then the centralServer
	frameNumber = 0
	personDetectionFrameThres = 10
	personDetectionFlag = False
	global global_count
	global cam1_list
	global cam2_list
	global central_list
	
	while True:
		frameNumber = frameNumber + 1
		print('*************************************')
		print('frameNumber: %d'%frameNumber)
		ok, cam1_image = cam1_video.read()
		if not ok:
			print('cam1_image not read!');
			break
		ok, cam2_image = cam2_video.read()
		if not ok:
			print('cam2_image not read!');
			break
		
		if(frameNumber == 1 or frameNumber%personDetectionFrameThres == 0):
			personDetectionFlag = True
		else:
			personDetectionFlag = False

		#cam1_personsTracked = cam_personTracking(cam1_image, cam1_tracker, personDetectionFlag)#True)#personDetectionFlag)
		#cam2_personsTracked = cam_personTracking(cam2_image, cam2_tracker, personDetectionFlag)#True)#personDetectionFlag)
		cam1_personsTracked, num_new_c1 = cam_personTracking(cam1_image, cam1_tracker, personDetectionFlag)#True)#personDetectionFlag)
		cam2_personsTracked, num_new_c2 = cam_personTracking(cam2_image, cam2_tracker, personDetectionFlag)#True)#personDetectionFlag)
		
		#print('CAMERA 1 INFO: ')
		#print(cam1_personsTracked)
		#print('Possible new person added this frame :',num_new_c1)
		
		#print('CAMERA 2 INFO: ')
		#print(cam2_personsTracked)
		#print('Possible new person added this frame :',num_new_c2)
		
		centralServer(cam1_personsTracked, cam2_personsTracked, cam1_image, cam2_image, num_new_c1, num_new_c2)
		print('central_list')
		print(central_list)
		print('cam1_list')
		print(cam1_list)
		print('cam2_list')
		print(cam2_list)
		print('*************************************')
		cam1_image_copy = deepcopy(cam1_image);
		cam2_image_copy = deepcopy(cam2_image);
		#cam1_image_copy2 = cam1_image[:];
		#cam2_image_copy2 = cam2_image[:];
		'''
		index = -1
		for newbox in cam1_personsTracked:
			index = index + 1
			p1 = (int(newbox[0]), int(newbox[1]))
			p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
			cv2.rectangle(cam1_image, p1, p2, (255,0,0), 2, 1)
			cv2.putText(cam1_image, "ID:"+str(index), p1, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
		
		index = -1
		for newbox in cam2_personsTracked:
			index = index + 1
			p1 = (int(newbox[0]), int(newbox[1]))
			p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
			cv2.rectangle(cam2_image, p1, p2, (255,0,0), 2, 1)
			cv2.putText(cam2_image, "ID:"+str(index), p1, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
		'''
		for id in central_list:
			if(cam1_list[id] != -1):
				newbox = cam1_personsTracked[cam1_list[id]]
				p1 = (int(newbox[0]), int(newbox[1]))
				p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
				cv2.rectangle(cam1_image_copy, p1, p2, (255,0,0), 2, 1)
				cv2.putText(cam1_image_copy, "ID:"+str(id), p1, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
				cv2.rectangle(cam1_image, p1, p2, (255,0,0), 2, 1)
				cv2.putText(cam1_image, "ID:"+str(cam1_list[id]), p1, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
			
			if(cam2_list[id] != -1):
				newbox = cam2_personsTracked[cam2_list[id]]
				p1 = (int(newbox[0]), int(newbox[1]))
				p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
				cv2.rectangle(cam2_image_copy, p1, p2, (255,0,0), 2, 1)
				cv2.putText(cam2_image_copy, "ID:"+str(id), p1, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
				cv2.rectangle(cam2_image, p1, p2, (255,0,0), 2, 1)
				cv2.putText(cam2_image, "ID:"+str(cam2_list[id]), p1, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
		
		cv2.imshow("Camera1", cam1_image)
		cv2.imshow("Camera2", cam2_image)
		
		cv2.imshow("Camera1_Central", cam1_image_copy)
		cv2.imshow("Camera2_Central", cam2_image_copy)
		
		keyPressed = cv2.waitKey(10)
		if(keyPressed == 27):
			break


def personDetection(image):
	#print('\t\tpersonDetection')
	#personsDetected = []
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
	(personsDetected, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
	#print(personsDetected)
	return list(personsDetected)

def personTracking(image, tracker): #Main Tracker Update (Update individual ID Here)
	#print('\t\tpersonTracking')
	ok, personsTracked = tracker.update(image)
	return personsTracked

'''
def  modifyTracker(tracker, persons):
	#Modify the persons list so that the new persons list () will contain only the newly added person at the end of the list. But the original list will be unmodified.
	#print('\t\tmodifyTracker')
	personsMod = []
	return personsMod
'''

def modifyTracker(tracker, persons, threshold_centroid, new):#tracked, detected, threshold_centroid
	#tracker contains 
	present = False
	newlist = list(tracker) #Convert Tuple to list
	for (x,y,w,h) in persons:#For every detected person in frame
		centroid_person = get_centroid(x,y,w,h)
		present = False
		for (xp,yp,wp,hp) in tracker: #For every tracker output
			centroid_tracker = get_centroid(xp,yp,wp,hp)
			if (check_threshold(centroid_tracker,centroid_person,threshold_centroid)):
				present = True
		if (present == False): #this is a new person
			newlist.append([x,y,w,h])	
			new +=1 #number of new persons added
	return newlist, new

def get_centroid(x,y,w,h):
	xmean = (2*x+w)/2
	ymean = (2*y+h)/2
	return xmean,ymean

def check_threshold(centroid_tracker,centroid_person,threshold_centroid):
	## taking euclidean distance
	distance = ((centroid_tracker[0]-centroid_person[0])**2+(centroid_tracker[1]-centroid_person[1])**2)**0.5
	if(distance < threshold_centroid):
		return True
	else:
		return False

###############################################################
def cam_personTracking(image, tracker, flag):
	#print('\tcam_personTracking')
	threshold_centroid = 60
	new = 0 #number of new persons added
	#Do tracking of currently tracked persons
	#personsTracked, personsTracked_ID = personTracking(image, tracker) #Main Tracker Update
	personsTracked = personTracking(image, tracker) #Main Tracker Update #LIST
	#print('\t\tpersonsTracked: %d'%len(personsTracked))
	#print(personsTracked)
	#If flag is True, detect if new persons are present and add a new tracker for those persons
	personsMod = personsTracked #Tuple now but later becomes List
	if(flag): #Every tenth frame
		personsDetected = personDetection(image) #Every 10th image detect all persons
		if personsDetected: #if frame not empty
			personsMod, new = modifyTracker(personsTracked, personsDetected, threshold_centroid, new)
			#print('\tAm I a List?: ')
			#print(isinstance(personsMod, list))
			#print('\t\tpersonsDetected: %d'%len(personsDetected));
			#print(personsDetected);
			#print('\t\tpersonsMod After: %d'%len(personsMod));
			#print(personsMod);
			for i in range(len(personsTracked),len(personsMod)):
				ok = tracker.add(cv2.TrackerBoosting_create(), image, tuple(personsMod[i]));
	
	return personsMod, new
	
def centralServer(cam1_persons, cam2_persons, cam1_image, cam2_image, num_new_c1, num_new_c2):
	global global_count
	global cam1_list
	global cam2_list
	global central_list
	threshold = 10 #threshold for sift feature match strength
	
	for i in range(num_new_c1):
		matchExists = False
		max_strength = 0
		match_idx = -1
		bBox1 = cam1_persons[len(cam1_persons) - i - 1]
		for j in range(len(cam2_list)):
			if(cam1_list[j] == -1):
				bBox2 = cam2_persons[cam2_list[j]]
				match_strength = check_image(cam1_image[int(bBox1[1]):int(bBox1[1]+bBox1[3]), int(bBox1[0]):int(bBox1[0]+bBox1[2])], cam2_image[int(bBox2[1]):int(bBox2[1]+bBox2[3]), int(bBox2[0]):int(bBox2[0]+bBox2[2])])
				if (match_strength > max_strength):
					max_strength = match_strength
					match_idx = j
		print('-----------max_strength: %g'%max_strength)
		if (max_strength > threshold):
			cam1_list[match_idx] = len(cam1_persons) - i - 1;
		else:
			cam1_list.append(len(cam1_persons) - i - 1)
			cam2_list.append(-1)
			central_list.append(global_count)
			global_count += 1
			
	for i in range(num_new_c2):
		matchExists = False
		max_strength = 0
		match_idx = -1
		bBox2 = cam2_persons[len(cam2_persons) - i - 1]
		for j in range(len(cam1_list)):
			if(cam2_list[j] == -1):
				bBox1 = cam1_persons[cam1_list[j]]
				match_strength = check_image(cam1_image[int(bBox1[1]):int(bBox1[1]+bBox1[3]), int(bBox1[0]):int(bBox1[0]+bBox1[2])], cam2_image[int(bBox2[1]):int(bBox2[1]+bBox2[3]), int(bBox2[0]):int(bBox2[0]+bBox2[2])])
				print('-----------max_strength: %g'%match_strength)
				if (match_strength > max_strength):
					max_strength = match_strength
					match_idx = j
		print('-----------max_strength: %g'%max_strength)
		if (max_strength > threshold):
			cam2_list[match_idx] = len(cam2_persons) - i - 1
		else:
			cam2_list.append(len(cam2_persons) - i - 1)
			cam1_list.append(-1)
			central_list.append(global_count)
			global_count += 1

#def isMatch()
	
def check_image(img_querry,img_train):
	##this function takes in two already imread objects and compare them and 
	## then send back the total features matched between the two images.
	img1 = img_querry
	img2 = img_train
	# Initiate SIFT detector
	#sift = cv2.SIFT()
	sift = cv2.xfeatures2d.SIFT_create()
	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)

	# FLANN parameters
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)   # or pass empty dictionary

	flann = cv2.FlannBasedMatcher(index_params,search_params)
	matches = flann.knnMatch(des1,des2,k=2)
	# matches to draw only good matches, so create a mask
	matchesMask = [[0,0] for i in range(len(matches))]

	# ratio test as per Lowe's paper
	count = 0
	for i,(m,n) in enumerate(matches):
	   if m.distance < 0.9*n.distance:#0.7*n.distance:
	       matchesMask[i]=[1,0]
	       count = count+1
	       
	return count
'''
		personsMod = list(tracker.getObjects())
	personsTracked = []
	if(flag):
		personsDetected = personDetection(image)
		print('\t\tpersonsDetected');
		print(personsDetected);
		print('\t\tpersonsMod Before')
		print(personsMod)
		#print('len(personsDetected): %d'%len(personsDetected))
		if personsDetected:
			personsMod = modifyTracker(personsMod, personsDetected, threshold_centroid)
		#Add a tracker for new person only!!!
		
		print('\t\tpersonsMod After');
		print(personsMod);
		for i in range(len(personsDetected),len(personsMod)):
			ok = tracker.add(cv2.TrackerMIL_create(), image, tuple(personsMod[i]));
		
		#personsMod = personsDetected
	if personsMod:
		print(personsMod);
		personsTracked = personTracking(image, tracker)
	return personsTracked
'''

if __name__ == '__main__' :
	arg1 = "Dataset\\campus4-c0.avi";
	arg2 = "Dataset\\campus4-c2.avi";
	main(arg1, arg2)

