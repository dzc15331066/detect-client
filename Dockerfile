FROM registry.cn-shenzhen.aliyuncs.com/paper-project/detect-client-base:latest
RUN pip install flask
WORKDIR home/detect-client
COPY . .
EXPOSE 9090
CMD ["python", "client/main.py"]
