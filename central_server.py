cam1_list = []
cam2_list = []
central_list = []
global_count = 0

def check_camera_number(cam_list,index_of_other_camera):

	for i in range(0,len(cam_list)):
		if(cam_list[i] == index_of_other_camera):
			return i

def update_index(index_of_other_camera,cam_new_list_number,index_of_cam_new_list_number):

	if(cam_new_list_number == 1 ):
		index = check_camera_number(cam2_list,index_of_other_camera)
		cam1_list[index] = index_of_cam_new_list_number
	else:
		index = check_camera_number(cam1_list,index_of_other_camera)
		cam2_list[index] = index_of_cam_new_list_number


def Book_keeping(cam_new_list_number,match,index_of_cam_new_list_number,index_of_other_camera):
	# cam_new_list_number : the camera where a new person came in ??
	# match: wether the same person is in camera 2 and 1 or not
	# index_of_cam_new_list_number: this is the corresponing index of the tracker which represent a person being tracked for camera number = cam_new_list_number
	# index_of_other_camera : index of the tracker with which the other camera person matches. e.g if a person moves in cam 1 who was already in cam 2 then this variables 
	# represent index of tracker representing the person in cam 2 	

	if(match):
		update_index(index_of_other_camera,cam_new_list_number,index_of_cam_new_list_number)

	else:
		if(cam_new_list_number == 1):
			cam1_list.append(index_of_cam_new_list_number)
			cam2_list.append(-1)
			central_list.append(global_count)
			global_count++
		else:
			cam2_list.append(index_of_cam_new_list_number)
			cam1_list.append(-1)
			central_list.append(global_count)
			global_count++



def remove_all(index):
	del cam1_list[index]
	del cam2_list[index]
	del central_list[index]


def remove_ID(cam_number,index):
	# this function assumes that if a person moves out of the caera vision. That particular camera will signal the central server to remove a entry with particular
	# index. so cam_number represent the camera and the index here gives the index used in the tracker of the camera.Central server will look for that index
	# and remove it from it completely if the same person donot exist in other vision otherwise it will keep one. Note that the camera will be called whenever the 
	# camera signals a change in number of person and indicating which particular person by referring to its corressponding index

	if cam_number == 1:
		for i in range(0,len(cam1_list)):
			if(cam1_list[i] == index):
				index = i 
				cam1_list[i] = -1
				break
			if(cam2_list[index] == -1 )
				remove_all(index)
	else:
		for i in range(0,len(cam2_list)):
			if(cam2_list[i] == index):
				index = i 
				cam2_list[i] = -1
				break
			if(cam2_list[index] == -1 )
				remove_all(index)


