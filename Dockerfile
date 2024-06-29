FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

WORKDIR /app

RUN apt-get update
RUN apt-get install -y dotnet-sdk-8.0

COPY pip-packages.txt pip-packages.txt
RUN pip install --upgrade pip
RUN pip install --upgrade -r pip-packages.txt
RUN rm ./pip-packages.txt

ENTRYPOINT /bin/sh -c "while sleep 1000; do :; done"