'''
for demo purpose a saved VIDEO_SOURCE is being used and being sent frame by
frame via API call to server.

Proposed solution for client side is to use pipelining technique, where there
will be 2 processors(pipeline stages) in the pipeline with following functions 
respectively:
1) Function to record/capture the frame
2) Function to send the frame to server via API call

A message queue can be used, if required, in case when frame capturing rate is
greater than the rate at which frames are being transmitted to the server. 

'''

#python frame_sender_client.py 

import skvideo.io
import requests
import json
VIDEO_SOURCE = "input.mp4"
url = 'http://127.0.0.1:8080/frame-receiver'
headers = {'Content-Type': 'application/json', 'Accept':'application/json'}
cap = skvideo.io.vreader(VIDEO_SOURCE,num_frames=1000)
cnt = 0
print('\nRequesting for "training model" and "creating Image Processing Pipline"\n')
res = requests.get(url,headers=headers)
print("Response code = ",res.status_code)
print('Training and Pipeline creation finished')

print('\n\nSending frames now.....\n\n')

for frame in cap:
	if not frame.any():
		break

	if cnt%20 == 0:

		data = {'frame': frame.tolist()}
		print('sending frame...')
		resp = requests.post(url,data=json.dumps(data), headers=headers)
		print("Response code = ", resp.status_code)
		print('frame sent\n\n')
	cnt+=1
