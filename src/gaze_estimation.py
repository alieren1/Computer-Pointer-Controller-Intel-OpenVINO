'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import os
import cv2
import math
from openvino.inference_engine import IECore, IENetwork

class Gaze_Estimation:
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

        self.input_name=[inp for inp in self.model.inputs.keys()]
        self.input_shape=self.model.inputs[self.input_name[1]].shape
        self.output_name=[out for out in self.model.outputs.keys()]  

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        core = IECore()

        self.exec_network = core.load_network(network=self.model, device_name=self.device)
        
        return self.exec_network

    def predict(self, image, eye_left, eye_right, eye_center, head_angle, disp):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        p_eye_left = self.preprocess_input(eye_left, self.input_shape)
        p_eye_right = self.preprocess_input(eye_right, self.input_shape)
        
        self.exec_network.start_async(request_id=0, inputs={'left_eye_image': p_eye_left,
                                                         'right_eye_image': p_eye_right,
                                                         'head_pose_angles': head_angle})
        
        if self.exec_network.requests[0].wait(-1) == 0:
            
            outputs = self.exec_network.requests[0].outputs[self.output_name[0]]
            
            out_image, gaze = self.preprocess_output(image, outputs, eye_center, disp)
                    
        return out_image, gaze

    def preprocess_input(self, image, input_shape):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        p_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        
        return p_frame

    def preprocess_output(self, image, outputs, eye_center, disp):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        gaze =  outputs[0]

        if(disp):
        
            eye_left_center_X = int(eye_center[0][0])
            eye_left_center_Y = int(eye_center[0][1])
            
            eye_right_center_X = int(eye_center[1][0])
            eye_right_center_Y = int(eye_center[1][1])
            
            cv2.arrowedLine(image, (eye_left_center_X, eye_left_center_Y), (eye_left_center_X + int(gaze[0] * 100), eye_left_center_Y + int(-gaze[1] * 100)), (255,0,255), 5)
            cv2.arrowedLine(image, (eye_right_center_X, eye_right_center_Y), (eye_right_center_X + int(gaze[0] * 100), eye_right_center_Y + int(-gaze[1] * 100)), (255,0,255), 5)

        return image, gaze
