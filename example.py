from LPRecogniser import LPRecogniser
import cv2
from utils import *
import argparse
import time
from pathlib import Path
lp_reco = LPRecogniser()
#write code detect license plate and character in video and calculate time and fps

def detect_lp_and_character_in_video(video_path, output_path):
    #load model
    lp_recognizer = LPRecogniser()
    #load video
    cap = cv2.VideoCapture(video_path)
    #get frame per second of video
    fps = cap.get(cv2.CAP_PROP_FPS)
    #get width and height of video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    #read until end of video
    while cap.isOpened():
        #read frame
        ret, frame = cap.read()
        if ret:
            #detect license plate
            lp_bboxes = lp_recognizer.lp_det.detect(frame)
            pred_results = lp_reco.predict(frame)
            bbox_only = [bbox for bbox, _ in pred_results]
            real_bboxes = recover_bbox(frame, bbox_only)
            label_only = [lab for _, lab in pred_results]
            print(str(label_only))
            #caulate time and fps
            # local variable 't0' referenced before assignment

            #draw bounding box and label
            for bbox, label in zip(real_bboxes, label_only):
                color = (0,255,255)
                draw_bbox(frame, label, yolo_to_bbox(frame, bbox), color, 2)
                #put fps and time
                # cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                # save image

                

            #write frame to output video  
            out.write(frame)  
         
        else:
            break
    #release VideoCapture and VideoWriter
    cap.release()
    out.release()

if __name__ == '__main__':
    #parse arguments
    parser = argparse.ArgumentParser()
    # path to video
    # "C:\Users\tamin\Downloads\Video\test_6.mp4"
    parser.add_argument('--video_path', type=str, default='C:/Users/tamin/Downloads/Video/test3.mp4')
    # path to output video
    # "C:/Users/tamin/Downloads/Video/test_6_output.mp4"
    parser.add_argument('--output_path', type=str, default='C:/Users/tamin/Downloads/Video/test_3_output.mp4')
    args = parser.parse_args()
    #check if video_path exists
    if not Path(args.video_path).exists():
        raise FileNotFoundError('video_path does not exist')
    #detect license plate and character in video
    detect_lp_and_character_in_video(args.video_path, args.output_path)



