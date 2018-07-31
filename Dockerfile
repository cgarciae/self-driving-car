FROM tensorflow/tensorflow:1.9.0-gpu-py3


RUN pip install numpy
RUN pip install pandas
RUN pip install dicto
RUN pip install click
RUN pip install tfinterface
RUN pip install dataget
RUN pip install python_path
RUN pip install python-socketio
RUN pip install eventlet

RUN apt-get update
RUN apt-get install -y netbase

RUN pip install flask

RUN apt-get install -y python3-tk

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8