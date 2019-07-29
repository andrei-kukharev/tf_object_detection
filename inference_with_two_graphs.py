#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Running two neural networks simultaneously.
The first NN is used for the prediction of an object presence (objectness).
The second NN is used for the prediction of the object location.
Platform: Raspberry Pi 3.
"""

import sys
import os
import argparse
import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
from PIL import Image
from time import sleep
import random
import io
from picamera import PiCamera
from picamera.array import PiRGBArray

PB1_PATH = '../pb/model_MobileNetV2_single-72-0.996-0.996[0.840].pb' 
PB2_PATH = PB1_PATH
INPUT_SIZE_1 = [3, 128, 128]
INPUT_SIZE_2 = [3, 128, 128]
INPUT_NODE_1 = 'input'
INPUT_NODE_2 = 'input'
OUTPUT_NODE_1 = 'output/Sigmoid'
OUTPUT_NODE_2 = 'output/Sigmoid' 
input_output_placeholders_1 = [INPUT_NODE_1 + ':0', OUTPUT_NODE_1 + ':0']
input_output_placeholders_2 = [INPUT_NODE_2 + ':0', OUTPUT_NODE_2 + ':0']

# Initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))


def get_image_as_array(image_file):
	""" Reading the image from a file and returning it as a numpy array.
	"""
	image = Image.open(image_file)
	shape = tuple(INPUT_SIZE[1:])
	image = image.resize(shape, Image.ANTIALIAS)
	image_as_array = np.array(image, dtype=np.float32) / 255.0
	return image_as_array


def image_to_array(image):
	""" Converting the image into a numpy array with normalization.
	"""
	return np.array(image, dtype=np.float32) / 255.0


def get_labels(labels_file):
	""" Read labels from a file.
	"""
	with open(labels_file) as f:
		labels = f.readlines()
		labels = [x.strip() for x in labels]
	return labels


def get_frozen_graph(pb_file):
	""" Loading a protobuf file and parsing it to an unserialized 
	computational graph
	"""
	with gfile.FastGFile(pb_file,'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
	return graph_def


def compress_graph_with_trt(graph_def, precision_mode):
	""" Compressing a computational graph using the TRT library.
	"""
	output_node = input_output_placeholders[1]

	if precision_mode == 0: 
		return graph_def

	trt_graph = trt.create_inference_graph(
		graph_def,
		[output_node],
		max_batch_size=1,
		max_workspace_size_bytes=2<<20,
		precision_mode=precision_mode)

	return trt_graph


def inference_from_camera_with_two_graphs(graph_def_1, graph_def_2):
	""" Capture images from the Raspberry's camera.

	Inputs
	------
	graph_def_1, graph_def_2 : computational graphs
	"""
	graph1 = tf.Graph()
	graph2 = tf.Graph()
	sess1 = tf.Session(graph=graph1)
	sess2 = tf.Session(graph=graph2)

	with graph1.as_default() as graph:
		inputs1, predictions1 =  tf.import_graph_def(graph_def_1, name='g1', 
			return_elements=input_output_placeholders_1)
			
	with graph2.as_default() as graph:
		inputs2, predictions2 =  tf.import_graph_def(graph_def_2, name='g2', 
			return_elements=input_output_placeholders_2)

	for i, frame in enumerate(camera.capture_continuous(\
							rawCapture, format="bgr", use_video_port=True)):
		# grab the raw NumPy array representing the image - this array
		# will be 3D, representing the width, height, and # of channels
		image_arr = frame.array
		image_cam = Image.fromarray(np.uint8(image_arr))				
		image1 = image_cam.resize(tuple(INPUT_SIZE_1[1:]), Image.ANTIALIAS)
		image1_arr = np.array(image1, dtype=np.float32) / 255.0
		if tuple(INPUT_SIZE_1[1:]) == tuple(INPUT_SIZE_2[1:]):
			image2_arr = image1_arr
		else:
			image2 = image_cam.resize(tuple(INPUT_SIZE_2[1:]), Image.ANTIALIAS)
			image2_arr = np.array(image2, dtype=np.float32) / 255.0

		pred_values1 = sess1.run(predictions1, feed_dict={inputs1: [image1_arr]})
		pred = pred_values1[0]
		pred_values2 = sess2.run(predictions2, feed_dict={inputs2: [image2_arr]})
		pred = pred_values2[0]

		sx, sy = image_cam.size
		x = pred[0] * sx
		y = pred[1] * sy
		w = pred[2] * sx
		h = pred[3] * sy
		box = (x, y, w, h)
		crop = image.crop(box)
		# Clear the stream in preparation for the next frame
		rawCapture.truncate(0)		
		sys.exit()


def createParser ():
	""" ArgumentParser
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', default=None, type=str,\
		help='input')
	parser.add_argument('-dir', '--dir', default="../images", type=str,\
		help='input')	
	parser.add_argument('-pb', '--pb', default="saved_model.pb", type=str,\
		help='input')
	parser.add_argument('-o', '--output', default="logs/1/", type=str,\
		help='output')
	return parser


if __name__ == '__main__':

	parser = createParser()
	arguments = parser.parse_args(sys.argv[1:])		
	pb_file = arguments.pb

	if arguments.input is not None:
		filenames = [arguments.input]
	else:
		filenames = []
		src_dir = arguments.dir
		listdir = os.listdir(src_dir)
		for filename in listdir:
			filenames.append(os.path.join(src_dir, filename))

	graph_def_1 = get_frozen_graph(PB1_PATH)
	graph_def_2 = get_frozen_graph(PB2_PATH)
	inference_from_camera_with_two_graphs(graph_def_1, graph_def_2)

