import eventlet
import socketio
import base64

import cv2
import numpy as np
import json
import os
from LPRecogniser import LPRecogniser
from utils import *
import base64
import io

import time 
lp_reco = LPRecogniser()
lp_recognizer = LPRecogniser()



sio = socketio.Server()

# base64_to_img
def base64_to_img(base64_string):
    imgdata = base64.b64decode(base64_string)
    return imgdata



@sio.on('connect')
def connect(sid, environ):
    print('Connected:', sid)

@sio.on('disconnect')
def disconnect(sid):
    print('Disconnected:', sid)

@sio.on('process_image')
def process_image(sid, data):
    # print('Received data:', data)
    # tinh time start
    t0 = time.time()
    img = base64_to_img(data)
    npimg = np.frombuffer(img, dtype=np.uint8)
    img = cv2.imdecode(npimg, 1)
    lp_bboxes = lp_recognizer.lp_det.detect(img)
    pred_results = lp_reco.predict(img)
    bbox_only = [bbox for bbox, _ in pred_results]
    real_bboxes = recover_bbox(img, bbox_only)
    label_only = [lab for _, lab in pred_results]
    for bbox, label in zip(real_bboxes, label_only):
        color = (255,255,0)
        draw_bbox(img, label, yolo_to_bbox(img, bbox), color, 1) 
    
    # convert image to base64 string utf-8 
    _, buffer = cv2.imencode('.jpg', img)
    jpg_as_text = base64.b64encode(buffer)
    jpg_as_text2 = jpg_as_text.decode('utf-8')
    with open('result.txt', 'w') as f:
        f.write(str(jpg_as_text2))
    # send message to client
    tst=''
    for i in label_only:
        tst=tst+i+' '
    # resize image 200*200
    # img = cv2.resize(img, (150, 150))
    # _, buffer = cv2.imencode('.jpg', img)
    # jpg_as_text_a = base64.b64encode(buffer)
    # jpg_as_text_a = jpg_as_text_a.decode('utf-8')

    # time end
    t1 = time.time()
    print('time: ', t1-t0, 's')
    # create json
    data = {
        'image': jpg_as_text2,
        'label': tst,
        'time_runing': t1-t0
    }


    
    sio.emit('processResult', data, room=sid)

if __name__ == '__main__':
    app = socketio.WSGIApp(sio)
    eventlet.wsgi.server(eventlet.listen(('localhost', 9999)), app)
