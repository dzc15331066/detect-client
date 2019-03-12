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


__HOST = '192.168.122.1'
__PORT = 8080


def encode(client,frame):
    #opencv image to jpg bytes
    r, buf = cv2.imencode(".jpg",frame)
    if r != True:
        return None
    print(len(buf))
    bytes_img = Image.fromarray(np.uint8(buf)).tobytes()
    print(type(bytes_img))

    print(len(bytes_img))
    fs = client.get_features(bytes_img)
    boxes = np.frombuffer(fs.boxes,dtype=np.int32)
    features = np.frombuffer(fs.features,dtype=np.float32)
    print(len(boxes))
    l = len(boxes)/4
    boxes = boxes.reshape(l,4)
    print(boxes)
    features = features.reshape(l,len(features)/l)
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

    transport.open()

   # deep_sort 
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True 
    
    video_capture = cv2.VideoCapture("client/person_detect.mp4")

    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output.avi', fourcc, 15, (w, h))
        
    fps = 0.0
    ##############################################
    loc_dic = {}
    in_count = 0 #in 计数器
    out_count = 0 #out 计数器
    ##############################################
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break;
        t1 = time.time()
        #############
        h, w, _ = frame.shape
        print(h,w)
        cv2.line(frame,(int(w/2),int(0)),(int(w/2),int(h)),(255,255,255))
        cv2.putText(frame,"in number:" + str(in_count), (10,40), 0, 1e-3 * h, (255,0,0),2)
        cv2.putText(frame,"out number:" + str(out_count), (10,60), 0, 1e-3 * h, (255,0,0),2)
        
        boxes, features = encode(client,frame) #image压缩为jpg格式，发送到gpu server进行yolov3检测，得到features后返回
        #print(features[0])#128
        tt1 = time.time()
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxes, features)]
        print(detections)
        
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        # Call the tracker
        
        tracker.predict()
        tracker.update(detections)
        print("used:",time.time()-tt1)
        
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
            
        #cv2.imshow('', frame)
            
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %f"%(fps))
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    if writeVideo_flag:
        out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
