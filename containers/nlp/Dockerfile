FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8


RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        git \
        wget \
        cmake \
        ninja-build \
        build-essential \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        python-is-python3 \
		libstdc++6 \
		gcc-multilib\
		libc6 \
		openssh-server \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/* 
	
ARG ssh_prv_key="$(cat C:/Users/eraye/.ssh/id_ed25519)"
ARG ssh_pub_key="$(cat C:/Users/eraye/.ssh/id_ed25519.pub)"

RUN mkdir -p /root/.ssh && \
    chmod 0700 /root/.ssh
	
COPY C:/Users/eraye/.ssh/known_hosts > /root/.ssh/known_hosts

RUN echo "$ssh_prv_key" > /root/.ssh/id_ed25519 && \
    echo "$ssh_pub_key" > /root/.ssh/id_ed25519.pub && \
    chmod 600 /root/.ssh/id_ed25519 && \
    chmod 600 /root/.ssh/id_ed25519.pub


RUN python3 -m pip install --upgrade pip && \
    python3 -m venv /opt/python3/venv/base
	


COPY requirements.txt /opt/python3/venv/base/
RUN /opt/python3/venv/base/bin/python3 -m pip install --no-cache-dir -r /opt/python3/venv/base/requirements.txt

#ENV CONDA_DIR /opt/conda
#RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
#    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
#ENV PATH=$CONDA_DIR/bin:$PATH

#COPY environment-complex.yml .
#RUN conda env update -f environment-complex.yml &&\
#    conda clean -tipy

COPY abstractive .
COPY extractive .


COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set entrypoint to bash
ENTRYPOINT ["/entrypoint.sh"]