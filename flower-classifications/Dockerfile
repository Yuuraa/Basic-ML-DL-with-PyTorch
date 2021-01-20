FROM pytorch/pytorch:latest

ADD train.py data requirements.txt ./

RUN pip install -r requirements.txt
RUN python3 train.py