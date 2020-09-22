# The following two commands can be used to build the Docker image and start a
# container:
#
# docker build -t michaelholtonprice/past_stoch_proc .
# docker run --name past_stoch_proc -it michaelholtonprice/past_stoch_proc
FROM ubuntu:18.04

# Set the following environmental variable to avoid interactively setting the
# timezone with tzdata when installing R
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y python3-pip && \
    apt-get install -y vim && \
    apt-get install -y git && \
    apt-get install -y apt-transport-https && \
    apt-get install -y software-properties-common && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9 && \
    add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/' && \
    apt-get update && \
    apt-get install -y r-base && \
    apt-get clean

# clone the repository
RUN git clone https://github.com/MichaelHoltonPrice/past_as_stoch_proc_code

# set the working directory to the code directory
WORKDIR /past_as_stoch_proc_code

RUN Rscript install_R_dependencies.R

# install python dependencies
RUN pip3 install -r requirements.txt

