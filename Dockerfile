FROM 763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-training:2.6.0-gpu-py312

COPY src/requirements-docker.txt /opt/program/
RUN pip install -r /opt/program/requirements-docker.txt