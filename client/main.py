#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('.')

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from example.Features import Client
from example.ttypes import FeaturesResult

import os
from timeit import time
import warnings
import cv2
import numpy as np
from PIL import Image

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection as ddet

from multiprocessing import Process, Manager
from flask import Flask, Response, render_template
app = Flask(__name__)

dt = Manager().dict()
dt["img"] = ""

__HOST = '127.0.0.1'
__PORT = 6300

last_stat_time = 0.0
bytes_send = 0
bytes_recv = 0

def encode(client,frame):
    global bytes_send
    global bytes_recv
    global last_stat_time
    #统计每秒收发的数据包大小
    now = time.time()
    if last_stat_time + 1 <= now:
        interval = now - last_stat_time
        print("send bytes/sec: %d" % (bytes_send/interval))
        print("recv bytes/sec: %d" % (bytes_recv/interval))
        bytes_send = 0
        bytes_recv = 0
        last_stat_time = now
    #opencv image to jpg bytes
    r, buf = cv2.imencode(".jpg",frame)
    if r != True:
        return None
    #print(len(buf))
    bytes_img = Image.fromarray(np.uint8(buf)).tobytes()
    #print(type(bytes_img))

    #print(len(bytes_img))
    fs = client.get_features(bytes_img)
    boxes = np.frombuffer(fs.boxes,dtype=np.int32)
    features = np.frombuffer(fs.features,dtype=np.float32)
    bytes_send += len(bytes_img)
    bytes_recv += len(fs.features) + len(fs.boxes)
    l = len(boxes)/4
    boxes = boxes.reshape(-1,4)
   
    features = features.reshape(l,-1)
    
    return boxes, features
def main():


   # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    tsocket = TSocket.TSocket(__HOST, __PORT)
    transport = TTransport.TFramedTransport(tsocket)
    protocol = TBinaryProtocol.TBinaryProtocol(transport)
    client = Client(protocol)
    extract_rate = 5 #抽帧频率
    transport.open()

   # deep_sort 
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    
    video_capture = cv2.VideoCapture("MOT16-09.mp4")
    frame_rate = video_capture.get(5)
    sample_interval = 1./ extract_rate
    print(frame_rate,extract_rate, sample_interval)
    delay = 1./frame_rate
    print(delay)
        
    fps = 0.0
    ##############################################
    loc_dic = {}
    in_count = 0 #in 计数器
    out_count = 0 #out 计数器
    ##############################################
    frame_count = 0
    global last_stat_time
    last_stat_time = time.time()
    w = 640
    h = 480
    last_sample_time = 0.0
    while True:
        start = time.time()
        ret, frame = video_capture.read()
        if ret != True:
            break;
        frame = cv2.resize(frame,(w, h))
        now = time.time()
        if last_sample_time + sample_interval <= now:
            print('jj')
            t1 = time.time()
            boxes, features = encode(client,frame) #image压缩为jpg格式，发送到gpu server进行yolov3检测，得到features后返回
            last_sample_time = time.time()
            nfps = 1./(time.time()-t1)
            if fps <= 0.1:
                fps = nfps
            else:
                fps = ( fps + nfps ) / 2
            print("detection fps= %f"%(fps))
            #print(features[0])#128
            tt1 = time.time()
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxes, features)]
            #print(detections)
            
            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
            
            # Call the tracker
            tracker.predict()
            tracker.update(detections)
            print("tracker used:",time.time()-tt1)
            
            
            for track in tracker.tracks:
                #print(track.track_id)
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)
                id_num = str(track.track_id)
                if id_num in loc_dic:
                    #判断上一帧运动轨迹
                    #向右运动，且经过分界线
                    last_x = loc_dic[id_num]
                    if bbox[0] > last_x and (bbox[0] > float(w/2) and last_x < float(w/2)):
                        print("##################in one#################")
                        loc_dic[id_num] = bbox[0]
                        in_count += 1
                    #向左移动，且经过分界线
                    elif bbox[0] < last_x and (bbox[0] < float(w/2) and last_x > float(w/2)):
                        print("###################out one################")
                        loc_dic[id_num] = bbox[0]
                        out_count += 1
                else:
                    loc_dic[id_num] = bbox[0]
            for det in detections:
                bbox = det.to_tlbr()
                cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
        frame_count += 1
                
        cv2.line(frame,(int(w/2),int(0)),(int(w/2),int(h)),(255,255,255))
        cv2.putText(frame,"in number:" + str(in_count), (10,40), 0, 1e-3 * h, (255,0,0),2)
        cv2.putText(frame,"out number:" + str(out_count), (10,60), 0, 1e-3 * h, (255,0,0),2)
        ret, frame = cv2.imencode('.jpg', frame)
        dt["img"] = frame.tobytes()
        wait_time = delay-(time.time()-start)
        print(wait_time)
        if wait_time > 0:
            time.sleep(wait_time)

    video_capture.release()

def get_frame():
    w = 640
    h = 480
    time.sleep(0.06)
    while True:
        jpeg = dt["img"]
        print(len(jpeg))
        if len(jpeg) == 0:
            continue
        return jpeg
def gen():
    while True:
        #start = time.time()
        img  = get_frame()
        #print(time.time() - start)
        yield ('--frame\r\n'
        'Content-Type: image/jpeg\r\n\r\n'+ img)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
  res = Response(gen(), mimetype='multipart/x-mixed-replace;boundary=frame')
  return res

if __name__ == '__main__':
    p = Process(target=main,args=())
    p.start()
    app.run(host="0.0.0.0",port=9090)
    p.terminate()
