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
LEFT_LANE_END = 640
RIGHT_LANE_START = 730


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



class ContourDetection(PipelineProcessor):
    '''
        Detecting moving objects.
        Purpose of this processor is to subtract background, get moving objects
        and detect them with a cv2.findContours method, and then filter off-by
        width and height.
        bg_subtractor - background subtractor isinstance.
        min_contour_width - min bounding rectangle width.
        min_contour_height - min bounding rectangle height.
        save_image - if True will save detected objects mask to file.
        image_dir - where to save images(must exist).
    '''
    @staticmethod
    def get_centroid(x, y, w, h):
        x1 = int(w / 2)
        y1 = int(h / 2)

        cx = x + x1
        cy = y + y1

        return (cx, cy)

    @staticmethod
    def save_frame(frame, file_name, flip=True):
        # flip BGR to RGB
        if flip:
            cv2.imwrite(file_name, np.flip(frame, 2))
        else:
            cv2.imwrite(file_name, frame)

    def __init__(self, bg_subtractor, min_contour_width=50, min_contour_height=50,
    max_contour_width=250, max_contour_height=350,
     save_image=False, image_dir='images'):
        super(ContourDetection, self).__init__()

        self.bg_subtractor = bg_subtractor
        self.min_contour_width = min_contour_width
        self.min_contour_height = min_contour_height
        self.max_contour_width = max_contour_width
        self.max_contour_height = max_contour_height
        self.save_image = save_image
        self.image_dir = image_dir

    def filter_mask(self, img, a=None):
        '''
            This filters are hand-picked just based on visual tests
        '''

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

        # Fill any small holes
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        # Remove noise
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

        # Dilate to merge adjacent blobs
        dilation = cv2.dilate(opening, kernel, iterations=2)

        return dilation

    def detect_vehicles(self, fg_mask, context):

        matches = []

        # finding external contours
        contours, hierarchy = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

        for (i, contour) in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour)
            contour_valid = (w >= self.min_contour_width) and (
                h >= self.min_contour_height) and (w <= self.max_contour_width) and (
                h <= self.max_contour_height)

            if not contour_valid:
                continue

            centroid = ContourDetection.get_centroid(x, y, w, h)

            matches.append(((x, y, w, h), centroid))

        return matches

    def __call__(self, context):
        frame = context['frame'].copy()
        frame_number = context['frame_number']

        fg_mask = self.bg_subtractor.apply(frame, None, 0.001)
        # just thresholding values
        fg_mask[fg_mask < 240] = 0
        fg_mask = self.filter_mask(fg_mask, frame_number)

        if self.save_image:
            ContourDetection.save_frame(fg_mask, os.path.join(settings.BASE_DIR, "fg_masks/mask_%04d.png" % frame_number), flip=False)

        context['objects'] = self.detect_vehicles(fg_mask, context)
        context['fg_mask'] = fg_mask

        return context


class SpeedDetection(PipelineProcessor):
    '''
        Detecting speed of moving vehicles.
    '''

    def __init__(self, save_image=False, image_dir='images', height=720, width=1280, frameCounter=0):
        super(SpeedDetection, self).__init__()

        self.carCascade = cv2.CascadeClassifier(os.path.join(settings.BASE_DIR, 'myhaar.xml'))
        self.bikeCascade = cv2.CascadeClassifier(os.path.join(settings.BASE_DIR, 'bikehaar.xml'))
        self.save_image = save_image
        self.image_dir = image_dir
        self.height = height
        self.width = width
        self.frameCounter = frameCounter

    @staticmethod
    def estimateSpeed(location1, location2):
        d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
        # ppm = location2[2] / carWidht
        ppm = 8.8
        d_meters = d_pixels / ppm
        #print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
        fps = 18
        speed = d_meters * fps * 3.6
        return speed

    def track_vehicles(self, image):

        rectangleColor = (0, 255, 0)
        currentCarID = 0
        fps = 0
        carTracker = {}
        carNumbers = {}
        carLocation1 = {}
        carLocation2 = {}
        speed = [None] * 1000

        start_time = time.time()
        # print(dir(image))
        # print(image.size)
        # print(image.shape)
        # input('hajjjj')

        image = cv2.resize(image, (self.width, self.height))
        resultImage = image.copy()

        carIDtoDelete = []

        for carID in carTracker.keys():
            trackingQuality = carTracker[carID].update(image)

            if trackingQuality < 7:
                carIDtoDelete.append(carID)

        for carID in carIDtoDelete:
            print ('Removing carID ' + str(carID) + ' from list of trackers.')
            print ('Removing carID ' + str(carID) + ' previous location.')
            print ('Removing carID ' + str(carID) + ' current location.')
            carTracker.pop(carID, None)
            carLocation1.pop(carID, None)
            carLocation2.pop(carID, None)

        image = image.astype('uint8')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cars = self.carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))

        cv2.rectangle(resultImage, (150, 250), (850, 350), (0, 255, 255), 4)

        for (_x, _y, _w, _h) in cars:
            x = int(_x)
            y = int(_y)
            w = int(_w)
            h = int(_h)

            x_bar = x + 0.5 * w
            y_bar = y + 0.5 * h

            matchCarID = None

            for carID in carTracker.keys():
                trackedPosition = carTracker[carID].get_position()

                t_x = int(trackedPosition.left())
                t_y = int(trackedPosition.top())
                t_w = int(trackedPosition.width())
                t_h = int(trackedPosition.height())

                t_x_bar = t_x + 0.5 * t_w
                t_y_bar = t_y + 0.5 * t_h

                if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                    matchCarID = carID

            if matchCarID is None:
                print ('Creating new tracker ' + str(currentCarID))

                tracker = dlib.correlation_tracker()
                tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))

                carTracker[currentCarID] = tracker
                carLocation1[currentCarID] = [x, y, w, h]

                currentCarID = currentCarID + 1

        for carID in carTracker.keys():
            trackedPosition = carTracker[carID].get_position()

            t_x = int(trackedPosition.left())
            t_y = int(trackedPosition.top())
            t_w = int(trackedPosition.width())
            t_h = int(trackedPosition.height())

            cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)

            # speed estimation
            carLocation2[carID] = [t_x, t_y, t_w, t_h]

        end_time = time.time()

        if not (end_time == start_time):
            fps = 1.0/(end_time - start_time)

        #cv2.putText(resultImage, 'FPS: ' + str(int(fps)), (620, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        for i in carLocation1.keys():
            [x1, y1, w1, h1] = carLocation1[i]
            [x2, y2, w2, h2] = carLocation2[i]

            # print 'previous location: ' + str(carLocation1[i]) + ', current location: ' + str(carLocation2[i])
            carLocation1[i] = [x2, y2, w2, h2]

            # print 'new previous location: ' + str(carLocation1[i])
            if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                # print('###############')
                # print([x1, y1, w1, h1], [x2, y2, w2, h2])

                if (speed[i] == None or speed[i] == 0) and y1 >= 275 and y1 <= 285:
                    speed[i] = self.estimateSpeed([x1, y1, w1, h1], [x2, y2, w2, h2])
                #if y1 > 275 and y1 < 285:
                if speed[i] != None and y1 >= 180:
                    cv2.putText(resultImage, str(int(speed[i])) + " km/hr", (int(x1 + w1/2), int(y1-5)),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        return resultImage

    def __call__(self, context):

        frame = context['frame'].copy()
        context['speed_frame'] = self.track_vehicles(frame)

        return context


class FramePusher(PipelineProcessor):
    '''
        Pushes frames towards client
    '''

    def push_frame_and_data(self,frame,op):
        IMAGE_DIR = os.path.join(settings.BASE_DIR, "static_content/detector_temp/")
        uid = uuid.uuid4()
        cv2.imwrite('{IMAGE_DIR}{n}.png'.format(IMAGE_DIR=IMAGE_DIR,n=uid),frame)
        settings.PUSHER_CLIENT.trigger('my-channel', 'my-event',
        {'frame_url': '/static/detector_temp/{n}.png'.format(n=uid),
         'op_data': str(op),
        })

    def __call__(self, context):

        img = context['speed_frame'].copy()
        img = img.astype(np.int32)
        c = 0
        left_lane_cnt = 0
        right_lane_cnt = 0

        for tup,centroid in context['objects']:
            if centroid[0] <= LEFT_LANE_END:
                left_lane_cnt+=1
            else:
                right_lane_cnt+=1

            x,y,w,h = tup
            # cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)
        op = "Left Lane Vehicle Count: {lcnt} ||  Right Lane Vehicle Count: {rcnt}".format(
                lcnt=left_lane_cnt,rcnt=right_lane_cnt)
        print(op)
        self.push_frame_and_data(img, op)
        context['op_data'] = op
        return context



