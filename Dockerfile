FROM tensorflow/tensorflow:1.9.0-gpu-py3


RUN pip install numpy
RUN pip install pandas
RUN pip install dicto
RUN pip install click
RUN pip install tfinterface
RUN pip install dataget
