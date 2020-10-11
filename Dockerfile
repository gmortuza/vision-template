FROM tensorflow/tensorflow:2.3.1-gpu

COPY . /home/project/

RUN cd /home/project && pip install -r requirements.txt

WORKDIR /home/project