import cv2
import dlib
import math
import os
import random
import skvideo.io
import time
import uuid

import numpy as np
import matplotlib.pyplot as plt

from django.conf import settings


# without this some strange errors happen
cv2.ocl.setUseOpenCL(False)
random.seed(123)

# ============================================================================
SHAPE = (720, 1280)  # HxW


class PipelineRunner(object):
    '''
        Very simple pipline.
        Just run passed processors in order with passing context from one to
        another.
    '''

    def __init__(self, pipeline=None):
        self.pipeline = pipeline or []
        self.context = {}

    def set_context(self, data):
        self.context = data

    def add(self, processor):
        if not isinstance(processor, PipelineProcessor):
            raise Exception(
                'Processor should be an isinstance of PipelineProcessor.')
        self.pipeline.append(processor)

    def remove(self, name):
        for i, p in enumerate(self.pipeline):
            if p.__class__.__name__ == name:
                del self.pipeline[i]
                return True
        return False

    def run(self):
        for p in self.pipeline:
            self.context = p(self.context)

        return self.context


class PipelineProcessor(object):
    '''
        Base class for processors.
    '''

    def __init__(self):
        pass


class SpeedDetection(PipelineProcessor):
    '''
        Detecting speed of moving vehicles.
    '''

    def __init__(self, save_image=False, image_dir='images', height=720, width=1280, frameCounter=0):
        super(SpeedDetection, self).__init__()

        self.carCascade = cv2.CascadeClassifier(
            os.path.join(settings.BASE_DIR, 'myhaar.xml'))
        self.save_image = save_image
        self.image_dir = image_dir
        self.height = height
        self.width = width
        self.frameCounter = frameCounter
        # tlx = top left x, brx = bottom right x
        self.roi_tlx = 180
        self.roi_tly = 270
        self.roi_brx = 850
        self.roi_bry = 440
        self.rectangleColor = (0, 255, 0)
        self.currentCarID = 0
        self.carTracker = {}
        self.car_speed = {}
        # carLocation1 = {}
        # carLocation2 = {}
        self.carRoiFrame = {}
        self.ppm = None


    def estimateSpeed(self, frame_diff, pixels_covered):
        # d_pixels = math.sqrt(math.pow(
        #     location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
        # ppm = location2[2] / carWidht
        #ppm = 8.8
        d_meters = pixels_covered / self.ppm
        #print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
        fps = 18
        time  = frame_diff / fps
        speed = (d_meters * 3.6) / time
        return speed

    def track_vehicles(self, image, frame_counter):

        # import ipdb
        # ipdb.set_trace()
        # carLocation1 = {}
        # carLocation2 = {}
        # speed = [None] * 1000

        # start_time = time.time()
        # print(dir(image))
        # print(image.size)
        # print(image.shape)
        # input('hajjjj')

        image = cv2.resize(image, (self.width, self.height))


        # image = image[self.roi_tly-100:self.roi_bry+220,self.roi_tlx+40:self.roi_brx-50]
        resultImage = image.copy()

        carIDtoDelete = []

        image = image.astype('uint8')

        for carID in self.carTracker.keys():
            trackingQuality = self.carTracker[carID].update(image)

            if trackingQuality < 7:
                carIDtoDelete.append(carID)

        for carID in carIDtoDelete:
            print ('Removing carID ' + str(carID) + ' from list of trackers.')
            print ('Removing carID ' + str(carID) + ' previous location.')
            print ('Removing carID ' + str(carID) + ' current location.')
            # self.carTracker.pop(carID, None)
            # self.car_speed.pop(carID, None)
            # carLocation1.pop(carID, None)
            # carLocation2.pop(carID, None)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cars = self.carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))

        cv2.rectangle(resultImage, (self.roi_tlx, self.roi_tly), (self.roi_brx, self.roi_bry), (0, 255, 255), 4)
        # cv2.rectangle(resultImage, (self.roi_tlx-50, self.roi_tly-100), (self.roi_brx+40, self.roi_bry+220), (240, 150, 55), 4)

        for (_x, _y, _w, _h) in cars:
            x = int(_x)
            y = int(_y)
            w = int(_w)
            h = int(_h)

            if self.ppm is None:
                self.ppm = w/2.4

            x_bar = x + 0.5 * w
            y_bar = y + 0.5 * h

            matchCarID = None

            for carID in self.carTracker.keys():
                trackedPosition = self.carTracker[carID].get_position()

                t_x = int(trackedPosition.left())
                t_y = int(trackedPosition.top())
                t_w = int(trackedPosition.width())
                t_h = int(trackedPosition.height())

                t_x_bar = t_x + 0.5 * t_w
                t_y_bar = t_y + 0.5 * t_h

                if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                    matchCarID = carID
                    print('*&**&*^&^&^&^***&')

            if matchCarID is None:
                print ('Creating new tracker ' + str(self.currentCarID))

                tracker = dlib.correlation_tracker()
                tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))

                self.carTracker[self.currentCarID] = tracker
                # carLocation1[currentCarID] = [x, y, w, h]

                self.currentCarID = self.currentCarID + 1


        # print(self.carRoiFrame.keys())
        # print("@@")
        # print(frame_counter)
        for carID in self.carTracker.keys():
            trackedPosition = self.carTracker[carID].get_position()


            t_x = int(trackedPosition.left())
            t_y = int(trackedPosition.top())
            t_w = int(trackedPosition.width())
            t_h = int(trackedPosition.height())
            # cv2.putText(resultImage, str(carID), (t_x, t_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)


            t_x_bar = t_x + 0.5 * t_w
            t_y_bar = t_y + 0.5 * t_h

            if t_y_bar >= self.roi_tly and t_y_bar <= self.roi_bry and carID not in self.carRoiFrame.keys():
                self.carRoiFrame[carID] = frame_counter
            if t_y_bar >= self.roi_bry and carID in self.carRoiFrame.keys():
                print("#####################################")
                cur_frame = frame_counter
                frame_diff = cur_frame - self.carRoiFrame[carID]
                pixels_covered = self.roi_bry - self.roi_tly
                self.car_speed[carID] = self.estimateSpeed(frame_diff, pixels_covered)
                cv2.putText(resultImage, str(int(self.car_speed[carID])) + " km/hr", (int(t_x + t_w / 2), int(
                    t_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                del self.carRoiFrame[carID]

            # cv2.rectangle(resultImage, (t_x, t_y),
            #               (t_x + t_w, t_y + t_h), self.rectangleColor, 4)


            # if self.car_speed.get(carID, None) is not None:
            #     cv2.putText(resultImage, str(int(self.car_speed[carID])) + " km/hr", (int(t_x + t_w / 2), int(
            #         t_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)


            # speed estimation
            # carLocation2[carID] = [t_x, t_y, t_w, t_h]

        # end_time = time.time()

        # if not (end_time == start_time):
        #     fps = 1.0/(end_time - start_time)

        #cv2.putText(resultImage, 'FPS: ' + str(int(fps)), (620, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # for i in carLocation1.keys():
        #     [x1, y1, w1, h1] = carLocation1[i]
        #     [x2, y2, w2, h2] = carLocation2[i]

        #     # print 'previous location: ' + str(carLocation1[i]) + ', current location: ' + str(carLocation2[i])
        #     carLocation1[i] = [x2, y2, w2, h2]

        #     # print 'new previous location: ' + str(carLocation1[i])
        #     if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
        #         # print('###############')
        #         # print([x1, y1, w1, h1], [x2, y2, w2, h2])

        #         if (speed[i] == None or speed[i] == 0) and y1 >= 275 and y1 <= 285:
        #             speed[i] = self.estimateSpeed(
        #                 [x1, y1, w1, h1], [x2, y2, w2, h2])
        #         # if y1 > 275 and y1 < 285:
        #         if speed[i] != None and y1 >= 180:
        #             cv2.putText(resultImage, str(int(speed[i])) + " km/hr", (int(x1 + w1 / 2), int(
        #                 y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        return resultImage

    def __call__(self, context):

        frame = context['frame'].copy()
        frame_counter = context['frame_counter']
        context['speed_frame'] = self.track_vehicles(frame, frame_counter)

        return context


class FramePusher(PipelineProcessor):
    '''
        Pushes frames towards client
    '''

    def push_frame_and_data(self, frame, op):
        IMAGE_DIR = os.path.join(
            settings.BASE_DIR, "static_content/detector_temp/")
        uid = uuid.uuid4()
        cv2.imwrite('{IMAGE_DIR}{n}.png'.format(
            IMAGE_DIR=IMAGE_DIR, n=uid), frame)
        settings.PUSHER_CLIENT.trigger('my-channel', 'my-event',
                                       {'frame_url': '/static/detector_temp/{n}.png'.format(n=uid),
                                        'op_data': str(op),
                                        })

    def __call__(self, context):

        img = context['speed_frame'].copy()
        img = img.astype(np.int32)
        c = 0

        # for tup,centroid in context['objects']:
        #     if centroid[0] <= LEFT_LANE_END:
        #         left_lane_cnt+=1
        #     else:
        #         right_lane_cnt+=1

        #     x,y,w,h = tup
        #     # cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)
        # op = "Left Lane Vehicle Count: {lcnt} ||  Right Lane Vehicle Count: {rcnt}".format(
        #         lcnt=left_lane_cnt,rcnt=right_lane_cnt)
        # print(op)
        op = ""
        self.push_frame_and_data(img, op)
        context['op_data'] = op
        return context
