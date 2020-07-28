'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import os
import cv2
from openvino.inference_engine import IECore, IENetwork

class Facial_Landmarks_Detection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.device = device
        self.model = model_name
        self.model_structure = model_name
        self.model_weights = os.path.splitext(self.model_structure)[0] + ".bin"

        try:
            self.model = IENetwork(self.model_structure, self.model_weights)
        except Exception:
            raise ValueError("Check your model path!")

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        core = IECore()
	
        self.exec_network = core.load_network(network=self.model, device_name=self.device)
        
        return self.exec_network

    def predict(self, image, face, face_coords, disp):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        p_frame = self.preprocess_input(face, self.input_shape)

        self.exec_network.start_async(request_id=0, inputs={self.input_name: p_frame})

        if self.exec_network.requests[0].wait(-1) == 0:
            
            outputs = self.exec_network.requests[0].outputs[self.output_name]
        
            image, eye_left, eye_right, eye_center = self.preprocess_output(outputs, face_coords, image, disp)
        
        return image, eye_left, eye_right, eye_center 

    def preprocess_input(self, image, input_shape):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        p_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        
        return p_frame

    def preprocess_output(self, outputs, face_coords, image, disp):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        landmarks = outputs.reshape(1, 10)[0]

        height = face_coords[3] - face_coords[1]
        width = face_coords[2] - face_coords[0]
        
        x_left = int(landmarks[0] * width) 
        y_left = int(landmarks[1] * height)
        
        xmin_left = face_coords[0] + x_left - 25
        ymin_left = face_coords[1] + y_left - 25
        xmax_left = face_coords[0] + x_left + 25
        ymax_left = face_coords[1] + y_left + 25
         
        x_right = int(landmarks[2] * width)
        y_right = int(landmarks[3] * height)
        
        xmin_right = face_coords[0] + x_right - 25
        ymin_right = face_coords[1] + y_right - 25
        xmax_right = face_coords[0] + x_right + 25
        ymax_right = face_coords[1] + y_right + 25
        
        if(disp):
            cv2.rectangle(image, (xmin_left, ymin_left), (xmax_left, ymax_left), (0,255,255), 3)        
            cv2.rectangle(image, (xmin_right, ymin_right), (xmax_right, ymax_right), (0,255,255), 3)
        
        eye_left_center =[face_coords[0] + x_left, face_coords[1] + y_left]
        eye_right_center = [face_coords[0] + x_right , face_coords[1] + y_right]      
        eye_center = [eye_left_center, eye_right_center]
        
        # Crop the left eye from the image
        eye_left = image[ymin_left:ymax_left, xmin_left:xmax_left]
        
        # Crop the right eye from the image
        eye_right = image[ymin_right:ymax_right, xmin_right:xmax_right]
        
        return image, eye_left, eye_right, eye_center
