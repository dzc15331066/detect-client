FROM registry.cn-shenzhen.aliyuncs.com/paper-project/detect-client-base:latest
WORKDIR home/detect-client
COPY . .
CMD ["python", "client/main.py"]
