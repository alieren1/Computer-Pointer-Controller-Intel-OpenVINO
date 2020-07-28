import sys
import cv2
import time
import logging as log

from argparse import ArgumentParser
from input_feeder import InputFeeder

from face_detection import Face_Detection
from head_pose_estimation import Head_Pose_Estimation
from facial_landmarks_detection import Facial_Landmarks_Detection
from gaze_estimation import Gaze_Estimation
from mouse_controller import MouseController

def build_argparser():

    parser = ArgumentParser()
    parser.add_argument("-fd", "--face_detection_model", required=True, type=str,
                        help="Path to an .xml file of the model Face Detection.")
   
    parser.add_argument("-hp", "--head_pose_model", required=True, type=str,
                        help="Path to an .xml file of the model Head Pose Estimation.")
                        
    parser.add_argument("-fl", "--facial_landmarks_model", required=True, type=str,
                        help="Path to an .xml file of the model Facial Landmarks Detection.")
                        
    parser.add_argument("-ge", "--gaze_estimation_model", required=True, type=str,
                        help="Path to an .xml file of the model Gaze Estimation.")
                       
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="CAM or path to image or video file.")

    parser.add_argument("-d", "--device", required=False, default="CPU", type=str,
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD.")
    
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.6,
                        help="Probability threshold for detection filtering"
                        "(0.6 by default)")

    parser.add_argument("-dsp", "--display", required=False, default=True, type=str,
                        help="Display the outputs of the models")
    
    parser.add_argument("-mp", "--mouse_precision", required=False, default='high', type=str,
                        help="Set the precision for mouse movement: high, medium, low.")
                        
    parser.add_argument("-spd", "--mouse_speed", required=False, default='fast', type=str,
                        help="Set the speed for mouse movement: fast, medium, slow.")
                        
    return parser

def handle_input(input_stream):
    
    if input_stream.endswith('.jpg') or input_stream.endswith('.png') or input_stream.endswith('.bmp'):
        input_type = 'image'
        
    elif input_stream == 'CAM':
        input_type = 'cam'
  
    elif input_stream.endswith('.mp4'):
        input_type = 'video'
    else: 
        log.error('Input type is not correct. Check for a valid input! .jpg, .png, .bmp, .mp4, CAM')
        sys.exit()    
    return input_type

def infer_on_stream(args):
    
    network_fd = Face_Detection(args.face_detection_model, args.device)
    network_hp = Head_Pose_Estimation(args.head_pose_model, args.device)
    network_fl = Facial_Landmarks_Detection(args.facial_landmarks_model, args.device)
    network_ge = Gaze_Estimation(args.gaze_estimation_model, args.device)
    
    mouse_cont = MouseController(args.mouse_precision, args.mouse_speed)
    
    starting_loading = time.time()
    
    network_fd.load_model()
    network_hp.load_model()
    network_fl.load_model()
    network_ge.load_model()
    
    duration_loading = time.time() - starting_loading
    
    input_type = handle_input(args.input)
    
    feed = InputFeeder(input_type=input_type, input_file=args.input)
    
    feed.load_data()
    
    starting_inference = time.time()
    
    for flag, frame in feed.next_batch():
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
         
        out_frame, face, face_coords = network_fd.predict(frame, args.prob_threshold, args.display)
        
        if len(face_coords) == 0:
            log.error("There is no face in the stream!")
            continue
            
        out_frame, head_angle = network_hp.predict(out_frame, face, face_coords, args.display)
        out_frame, eye_left, eye_right, eye_center = network_fl.predict(out_frame, face, face_coords, args.display)
        out_frame, gaze = network_ge.predict(out_frame, eye_left, eye_right, eye_center, head_angle, args.display)
        
        mouse_cont.move(gaze[0], gaze[1])
        
        if key_pressed == 27:
            break
       
        cv2.imshow('Visualization', cv2.resize(out_frame,(600,400)))
     
    duration_inference = time.time() - starting_inference
    
    print("Total loading time is: {}\nTotal inference time is: {} ".format(duration_loading, duration_inference))
    
    feed.close()
    cv2.destroyAllWindows
 
def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()

    #Perform inference on the input stream
    infer_on_stream(args)

if __name__ == '__main__':
    main()