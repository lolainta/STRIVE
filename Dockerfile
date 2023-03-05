FROM python:3.6

RUN pip3 install --upgrade pip

COPY . /strive
WORKDIR /strive
RUN pip3 install numpy==1.19.5
RUN pip3 install -r requirements.txt --no-compile

CMD sh

