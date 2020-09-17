FROM ubuntu:18.04

RUN apt-get update && \
    apt-get install -y python3-pip && \
    apt-get install -y vim && \
    apt-get clean

# set the working directory in the container
WORKDIR /code

# copy the dependencies file to the working directory
COPY requirements.txt .

# copy the dependencies file to the working directory
COPY seshat.py .

# install dependencies
RUN pip3 install -r requirements.txt

# docker build -t michaelholtonprice/seshat .
# docker run --name seshat -it michaelholtonprice/seshat
