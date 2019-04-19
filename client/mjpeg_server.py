from flask import Flask, Response, render_template
import cv2, os
from multiprocessing import Process, Manager
import time
app = Flask(__name__)

dt = Manager().dict()
dt["img"] = ""
def count():
    cap1 = cv2.VideoCapture("MOT16-09.mp4")
    frame_rate = cap1.get(5)
    delay = 1/frame_rate
    print("\tstart counting process ... ")
    while True:
        time.sleep(delay)
        success, img = cap1.read()
        if not success:
            return
            #cv2.imshow("Image",img)
        ret, jpeg = cv2.imencode(".jpg",img)
        dt["img"] = jpeg.tobytes()
    cap1.release()
    #cv2.destroyAllWindows()

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
    p = Process(target=count,args=())
    p.start()
    app.run(host="0.0.0.0",port=9090)
    p.terminate()
    #print("hh")