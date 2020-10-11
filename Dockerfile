FROM ubuntu:20.04

RUN apt-get update && apt-get install -y python python3-pip sudo
RUN pip3 install tensorflow==2.3.1

RUN alias python=python3
RUN alias pip=pip3

RUN useradd -m golam

RUN chown -R golam:golam /home/golam

COPY --chown=golam . /home/golam/

RUN cd /home/golam && pip3 install -r requirements.txt

USER golam


WORKDIR /home/golam
