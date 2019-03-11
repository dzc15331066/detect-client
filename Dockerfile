FROM debian:9.5
WORKDIR home/detect-client
COPY . .
RUN apt-get update && apt-get install -y python python-pip python-qt4&& pip install thrift opencv-python numpy pillow sklearn
CMD ["python", "client/main.py"]
