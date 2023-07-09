FROM ubuntu:22.04 as ubuntu

WORKDIR /root

COPY requirements.txt ./requirements.txt
COPY constants.py constants.py
COPY ingest.py ingest.py
COPY run_localGPT.py run_localGPT.py
COPY utils.py utils.py
COPY SOURCE_DOCUMENTS/ SOURCE_DOCUMENTS/

RUN	apt-get update -y && \
	apt-get install -y python3-pip && \
	pip3 install -r requirements.txt

CMD [ "/bin/bash" ]
