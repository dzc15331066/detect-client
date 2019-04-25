FROM registry.cn-shenzhen.aliyuncs.com/paper-project/detect-client-base:latest
RUN pip install flask
WORKDIR home/detect-client
COPY . .
CMD ["python", "client/main.py"]
