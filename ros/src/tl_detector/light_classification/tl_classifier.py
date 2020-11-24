import rospy
from styx_msgs.msg import TrafficLight
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
from glob import glob
import os
from keras.models import load_model

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        cwd = os.path.dirname(os.path.realpath(__file__))

        # load keras Lenet style model from file
        self.class_model = load_model(cwd+'/model.h5')
        self.class_graph = tf.get_default_graph()

        # detection graph
        self.dg = tf.Graph()
        # load 
        with self.dg.as_default():
            gdef = tf.GraphDef()
            with open(cwd+"/models/frozen_inference_graph.pb", 'rb') as f:
                gdef.ParseFromString( f.read() )
                tf.import_graph_def( gdef, name="" )

            #get names of nodes. from https://www.activestate.com/blog/2017/08/using-pre-trained-models-tensorflow-go
            self.session = tf.Session(graph=self.dg )
            self.image_tensor = self.dg.get_tensor_by_name('image_tensor:0')
            self.detection_boxes =  self.dg.get_tensor_by_name('detection_boxes:0')
            self.detection_scores = self.dg.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.dg.get_tensor_by_name('detection_classes:0')
            self.num_detections    = self.dg.get_tensor_by_name('num_detections:0')

        self.tlclasses = [ TrafficLight.RED, TrafficLight.YELLOW, TrafficLight.GREEN ]
        self.tlclasses_d = { TrafficLight.RED : "RED", TrafficLight.YELLOW:"YELLOW", TrafficLight.GREEN:"GREEN", TrafficLight.UNKNOWN:"UNKNOWN" }

        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        return TrafficLight.UNKNOWN
