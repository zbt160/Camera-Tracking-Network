import cv2
import imutils
import sys
 
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
		cam1_personsTracked, new_flag_1 = cam_personTracking(cam1_image, cam1_tracker, personDetectionFlag)#True)#personDetectionFlag)
		cam2_personsTracked, new_flag_2 = cam_personTracking(cam2_image, cam2_tracker, personDetectionFlag)#True)#personDetectionFlag)
		
		print('CAMERA 1 INFO: ')
		print(cam1_personsTracked)
		print('Possible new person added this frame :',new_flag_1)
		
		print('CAMERA 2 INFO: ')
		print(cam2_personsTracked)
		print('Possible new person added this frame :',new_flag_2)
		
		#aggregatedPersonsList = centralServer(cam1_personsTracked, cam2_personsTracked, cam1_image, cam2_image, new_flag_1, new_flag_2)
		aggregatedPersonsList = []
		if(len(aggregatedPersonsList)):
			#Draw tracks and bbox for individual camera outputs and aggregated outputs
			print('-------------------------------------------------------')
		print('*************************************')
		cam1_image_copy = cam1_image[:];
		cam2_image_copy = cam2_image[:];
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
		for person in aggregatedPersonsList:
			if(person.vis1):
				newbox = person.box1
				p1 = (int(newbox[0]), int(newbox[1]))
				p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
				cv2.rectangle(cam1_image_copy, p1, p2, (255,0,0), 2, 1)
				cv2.putText(cam1_image_copy, "ID:"+str(person.ID), p1, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
				
			if(person.vis2):
				newbox = person.box2
				p1 = (int(newbox[0]), int(newbox[1]))
				p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
				cv2.rectangle(cam2_image_copy, p1, p2, (255,0,0), 2, 1)
				cv2.putText(cam2_image_copy, "ID:"+str(person.ID), p1, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
		
		
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
			new = True
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
	threshold_centroid = 30
	new_flag = False
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
			personsMod, new_flag = modifyTracker(personsTracked, personsDetected, threshold_centroid, new_flag)
			#print('\tAm I a List?: ')
			#print(isinstance(personsMod, list))
			#print('\t\tpersonsDetected: %d'%len(personsDetected));
			#print(personsDetected);
			#print('\t\tpersonsMod After: %d'%len(personsMod));
			#print(personsMod);
			for i in range(len(personsTracked),len(personsMod)):
				ok = tracker.add(cv2.TrackerMIL_create(), image, tuple(personsMod[i]));
	
	return personsMod, new_flag
	
class person:
	def __init__(self):
		self.vis1 = None
		self.vis2 = None
		self.box1 = None
		self.box2 = None
		self.ID = None

def centralServer(cam1_persons, cam2_persons, cam1_image, cam2_image, c1_ID, c2_ID, new1, new2):
	threshold = 20 #threshold for sift feature match strength
	unique_persons = []
	matchidx2 = []
	global_ID = []
	if(len(cam1_persons) > 0):
		for i in range(len(cam1_persons)):
			max_strength = 0
			p = person()
			bBox1 = cam1_persons[i]
			#cv2.imshow("Cam1_cropped",cam1_image[int(bBox1[1]):int(bBox1[1]+bBox1[3]), int(bBox1[0]):int(bBox1[0]+bBox1[2])]);
			#cv2.waitKey(0);
			if(len(cam2_persons) > 0): #and ((new1 == True) or (new2 == True)):
				for j in range(len(cam2_persons)):
					bBox2 = cam2_persons[j]
					match_strength = check_image(cam1_image[int(bBox1[1]):int(bBox1[1]+bBox1[3]), int(bBox1[0]):int(bBox1[0]+bBox1[2])], cam2_image[int(bBox2[1]):int(bBox2[1]+bBox2[3]), int(bBox2[0]):int(bBox2[0]+bBox2[2])])
					if (match_strength > max_strength):
						max_strength = match_strength
						match_idx = j
						match_id = c2_ID[j]
			if (max_strength > threshold):
				matchidx2.append(match_idx)
				p.vis1 = True
				p.vis2 = True
				p.box1 = bBox1
				p.box2 = cam2_persons[match_idx]
				p.ID = id_count
				id_count +=1
			else:
				p.vis1 = True
				p.vis2 = False
				p.box1 = bBox1
				p.box2 = None
				p.ID = id_count
				id_count +=1
			unique_persons.append(p)
	leftOver2 = list(set(list(range(0,len(cam2_persons)))) - set(matchidx2)) 
	p = person()
	for idx in leftOver2:
		p.vis1 = False
		p.vis2 = True
		p.box1 = None
		p.box2 = cam2_persons[idx]
		p.ID = id_count
		id_count +=1
		unique_persons.append(p)
	
	return unique_persons
	
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
	   if m.distance < 0.7*n.distance:
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
	arg1 = "campus4-c0.avi";
	arg2 = "campus4-c2.avi";
	main(arg1, arg2)

