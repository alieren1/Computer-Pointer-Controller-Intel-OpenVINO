'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import cv2
from math import cos, sin, pi
from openvino.inference_engine import IECore, IENetwork

class Head_Pose_Estimation:
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
        self.output_name= [out for out in self.model.outputs.keys()]

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
            
            outputs = self.exec_network.requests[0].outputs
            out_image, head_angle  = self.preprocess_output(image, outputs, face_coords, disp)
            
        return out_image, head_angle

    def draw_out(self, image, head_angle ,face_coords): 

        y=head_angle[0]
        p=head_angle[1]
        r=head_angle[2]
        
       	xmin = face_coords[0]
        ymin = face_coords[1]
       	xmax = face_coords[2]
        ymax = face_coords[3]
        

        sin_y = sin(y * pi / 180)
       	cos_y = cos(y * pi / 180)
       	sin_p = sin(p * pi / 180)
        cos_p = cos(p * pi / 180)
        sin_r = sin(r * pi / 180)
        cos_r = cos(r * pi / 180)
        
        x = int((xmin + xmax) / 2)
        y = int((ymin + ymax) / 2)
        
        cv2.line(image, (x,y), (x+int(70*(cos_r*cos_y+sin_y*sin_p*sin_r)), y+int(70*cos_p*sin_r)), (0, 0, 255), thickness=3)
        cv2.line(image, (x, y), (x+int(70*(cos_r*sin_y*sin_p+cos_y*sin_r)), y-int(70*cos_p*cos_r)), (0, 255, 0), thickness=3)
        cv2.line(image, (x, y), (x + int(70*sin_y*cos_p), y + int(70*sin_p)), (255, 0, 0), thickness=3)
       
        return image

    def preprocess_input(self, image, input_shape):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        p_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

    def preprocess_output(self, image, outputs, face_coords, disp):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        y = outputs['angle_y_fc'][0][0]
        p = outputs['angle_p_fc'][0][0]
        r = outputs['angle_r_fc'][0][0]
        
        head_angle = [y, p, r]
        
        if (disp):
            out_image = self.draw_out(image, head_angle, face_coords)
        
        return out_image, head_angle
