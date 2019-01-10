
#Based on https://github.com/fizyr/keras-retinanet/blob/master/examples/ResNet50RetinaNet.ipynb 

import argparse
import os 
# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

def parse_args():
    parser = argparse.ArgumentParser(description='Test Retina Model with MIO or Bogota images')
    parser.add_argument(
        '--images', help="images directory", default='', type=str)
    parser.add_argument(
        '--dataset', help="MIO, Coco, Bogota", 
        default='', type=str)
    parser.add_argument(
        '--model', help="Retina model path", 
        default='', type=str)
    parser.add_argument(
        '--output', help="output dir",
        default='', type=str)
    parser.add_argument(
        '--gpu', help="output dir",
        default='', type=int)

    return parser.parse_args()
def get_mio():
	label_to_names = {0:'articulated_truck', 1 : 'bicycle', 2 : 'bus', 3: 'car',4: 'motorcycle',5: 'motorized_vehicle',6: 'non-motorized_vehicle',7: 'pedestrian',8: 'pickup_truck',9: 'single_unit_truck',10: 'work_van'}
	return label_to_names
def get_bogota():
	label_to_names = {0:'articulated_truck', 1 : 'bicycle', 2 : 'bus', 3: 'car',4: 'motorcycle',5: 'suv',6: 'taxi',7: 'pedestrian',8: 'pickup_truck',9: 'single_unit_truck',10: 'work_van'}
	return label_to_names

def operate_image(im):
		image = read_image_bgr(im)
		draw = image.copy()
		draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
		image = preprocess_image(image)
		image, scale = resize_image(image)
		return image, draw, scale 

def write_image(out, image):
		out = out + '.png'
		cv2.imwrite(out,image)

def read_images(images_dir,model,lbls,output):
	images = os.listdir(images_dir)
	for im in images:
		base = im
		im = os.path.join(images_dir,im)
		image, draw, scale =operate_image(im)

		# process image
		start = time.time()
		boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

		# correct for image scale
		boxes /= scale
		print("processing time: ", time.time() - start)
		# visualize detections
		for box, score, label in zip(boxes[0], scores[0], labels[0]):
    		# scores are sorted so we can break
			if score < 0.5:
				break
			label = label -1       
			color = label_color(label)
    
			b = box.astype(int)
			draw_box(draw, b, color=color)
    
			caption = "{} {:.3f}".format(lbls[label], score)
			draw_caption(draw, b, caption)
			out = os.path.join(output,base)
			write_image(out,draw)

def process_images(images_dir, model_path, lbls,output):
	# load retinanet model
	model = models.load_model(model_path, backbone_name='resnet50')
	read_images(images_dir,model,lbls,output)
def route_dataset_labels(dset):
	if (dset == 'bogota'):
		return get_bogota()
	elif (dset=='mio'):
		return get_mio()
	else:
		print('There is not a dataset asociated with the input')	
def main(args_):
	dataset = str(args_.dataset).lower()
	images_dir  = args.images 
	model_path = args.model
	print(args)
	output = args.output
	if args.gpu:
		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	if not os.path.exists(output):
			os.makedirs(output)
	lbls =   route_dataset_labels(dataset)
	process_images(images_dir, model_path, lbls,output)
if __name__=='__main__':
	args = parse_args()
	main(args)


