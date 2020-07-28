'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import os
import cv2
from openvino.inference_engine import IECore, IENetwork

class Face_Detection:
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
        self.output_shape = self.model.outputs[self.output_name].shape

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        core = IECore()

        self.exec_network = core.load_network(network=self.model, device_name=self.device)
        
        return self.exec_network

    def predict(self, image, prob_threshold, disp):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        p_frame = self.preprocess_input(image, self.input_shape)
	
        self.exec_network.start_async(request_id=0, inputs={self.input_name: p_frame})
        
        if self.exec_network.requests[0].wait(-1) == 0:
            
            outputs = self.exec_network.requests[0].outputs[self.output_name]
            
            out_image, face, face_coords = self.preprocess_output(image, outputs, prob_threshold, disp)
             
        return out_image, face, face_coords

    def preprocess_input(self, image, input_shape):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        p_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        return p_frame

    def preprocess_output(self, image, outputs, prob_threshold, disp):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        width, height = image.shape[1], image.shape[0]
           
        face_coords = []
        face = image
        for i in outputs[0][0]:
            confidence = i[2]
        
            if confidence >= prob_threshold:
                
                xmin = int(i[3] * width)
                ymin = int(i[4] * height)
                xmax = int(i[5] * width)
                ymax = int(i[6] * height)
                
                face_coords.append(xmin)
                face_coords.append(ymin)
                face_coords.append(xmax)
                face_coords.append(ymax)
                
                face = image[ymin:ymax, xmin:xmax]
                
                if(disp):
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,255,255), 3)   
        
        return image, face, face_coords
