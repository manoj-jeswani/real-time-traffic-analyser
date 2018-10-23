import os
import random

import numpy as np
import skvideo.io
import cv2
import matplotlib.pyplot as plt
import uuid
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
        im2, contours, hierarchy = cv2.findContours(
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
            ContourDetection.save_frame(fg_mask,"mask_%04d.png" % frame_number, flip=False)

        context['objects'] = self.detect_vehicles(fg_mask, context)
        context['fg_mask'] = fg_mask

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

        img = context['frame'].copy()
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
            cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)
        op = "Left Lane Vehicle Count: {lcnt} ||  Right Lane Vehicle Count: {rcnt}".format(
                lcnt=left_lane_cnt,rcnt=right_lane_cnt)    
        print(op)
#        self.push_frame_and_data(img,op)
        self.push_frame_and_data(context['frame'],op)
        context['op_data'] = op
        return context



