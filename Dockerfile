# 使用官方的Python 3.10作为基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 复制当前目录内容到容器的/app目录
COPY . /app

# 安装所需的包
RUN pip install --no-cache-dir -r requirements.txt

# 暴露8000端口
EXPOSE 8000

# 运行应用程序
CMD ["chainlit", "run", "app.py", "-w"]
