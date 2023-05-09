from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import cv2
import numpy as np
import json
import os
from LPRecogniser import LPRecogniser
from utils import *
import base64
import io
import random
import paho.mqtt.client as mqtt
import time 
lp_reco = LPRecogniser()
lp_recognizer = LPRecogniser()

app = FastAPI()


mqtt_broker = "45.251.112.69";
topic = "esp32/statuss";
topic2 = "parking";
mqtt_username = "ViotBroker";
mqtt_password = "Viot123!";
mqtt_port = 1104; 

client_id = f'python-mqtt-{random.randint(0, 1000)}'

def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt.Client(client_id)
    client.username_pw_set(mqtt_username, mqtt_password)
    client.on_connect = on_connect
    client.connect(mqtt_broker, mqtt_port)
    return client

client = connect_mqtt()
client.loop_start()

# base64_to_img
def base64_to_img(base64_string):
    imgdata = base64.b64decode(base64_string)
    return imgdata

def predict(img):
    lp_bboxes = lp_recognizer.lp_det.detect(img)
    pred_results = lp_reco.predict(img)
    bbox_only = [bbox for bbox, _ in pred_results]
    real_bboxes = recover_bbox(img, bbox_only)
    label_only = [lab for _, lab in pred_results]
    print(str(label_only))
            #caulate time and fps
            # local variable 't0' referenced before assignment

            #draw bounding box and label
    for bbox, label in zip(real_bboxes, label_only):
        color = (0,255,255)
        draw_bbox(img, label, yolo_to_bbox(img, bbox), color, 2)
            
    cv2.imwrite('result.jpg', img)

# show hello world
@app.get("/hi")
async def root():
    return {"message": "Hello World"}
    
# create websocket connection and send message to client
@app.websocket("/ha")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        # receive message from client
        data = await websocket.receive_text()
        # get data to json
        data = json.loads(data)
        # get txt from json
        txt = data['tst']
        B64 = data['base64']
        uid = data['uid']
        nameParking = data['nameParking']
        print(txt)
        # print(B64)

       
        # read image from client base64 string 
        img = base64_to_img(B64)
        # convert image to numpy array
        npimg = np.frombuffer(img, dtype=np.uint8)
        # convert numpy array to image
        img = cv2.imdecode(npimg, 1)
        # predict
        lp_bboxes = lp_recognizer.lp_det.detect(img)
        pred_results = lp_reco.predict(img)
        bbox_only = [bbox for bbox, _ in pred_results]
        real_bboxes = recover_bbox(img, bbox_only)
        label_only = [lab for _, lab in pred_results]
        # print(str(label_only))
        
            #caulate time and fps
            # local variable 't0' referenced before assignment

            #draw bounding box and label
        for bbox, label in zip(real_bboxes, label_only):
            color = (255,255,0)
            draw_bbox(img, label, yolo_to_bbox(img, bbox), color, 2) 
        
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
    #   crop image with bbox
        for bbox in real_bboxes:
            x1, y1, x2, y2 = yolo_to_bbox(img, bbox)
            crop_img = img[y1:y2, x1:x2]
            # convert crop_img to base64 string utf-8
            _, buffer = cv2.imencode('.jpg', crop_img)
            jpg_as_text3 = base64.b64encode(buffer)
            jpg_as_text4 = jpg_as_text3.decode('utf-8')

        # get time 
        now = time.localtime()
        time_now = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
      
        
        # await websocket.send_text(str(label_only))
        a={'tst':tst,'txt':txt,'id':5,'base64':jpg_as_text4,'uid':uid,'nameParking':nameParking,'time_now':time_now}
        
        b ={'tst':tst,'base64':jpg_as_text2}
        #  SEND JSON TO mqtt broker txt, jpg_as_text2,tst
        client.publish(topic, json.dumps(a))
       
        # get message from mqtt broker topic2 and show message print
        # client.subscribe(topic2)
        # stt = "s"
        # def on_message(client, userdata, msg):
        #     print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")
        #     #  read json from mqtt broker
        #     data = json.loads(msg.payload.decode())
        #     #  get stt from json
        #     stt = data['stt']
        #     #  send json to client
        #     if stt =='1':
        #         print(123)
          
        
        # client.on_message = on_message
       
        # print(stt)
        
        await websocket.send_text(json.dumps(b))
        # send image to client
        # await websocket.send_text(jpg_as_text)


        cv2.imwrite('result.jpg', img)
        # show image
        

    
        


    
    