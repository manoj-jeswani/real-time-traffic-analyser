import numpy
import uuid
import json
import cv2
import os
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from detector import trainer
from .image_processor import *

IMAGE_DIR = "/"


class FrameReceiver(APIView):
	MOG = None
	PIPELINE = None
	def get(self,request,format=None):
		# FrameReceiver.MOG = trainer.fun()
		print("\nCreating image processing pipeline...\n")
		# processing pipline for programming conviniance
		FrameReceiver.PIPELINE = PipelineRunner(pipeline=[
		SpeedDetection(),
		FramePusher(),
		# we use y_weight == 2.0 because traffic are moving vertically on video
		# use x_weight == 2.0 for horizontal.
		])
		return Response(None, status=status.HTTP_201_CREATED)

	def post(self, request, format=None):
		frame = request.data['frame']
		frame_counter = request.data['frame_counter']

		frame = numpy.array(frame)
		if isinstance(frame,numpy.ndarray):
			print(FrameReceiver.MOG)

			FrameReceiver.PIPELINE.set_context({
			'frame': frame,
			'frame_number': uuid.uuid4(),
			'frame_counter': frame_counter,
			})
			ctx = FrameReceiver.PIPELINE.run()
			return Response(None, status=status.HTTP_201_CREATED)

		return Response(None, status=status.HTTP_400_BAD_REQUEST)


def index_view(request):

	ctx = {

	}

	return render(request, 'base/index.html',ctx)

