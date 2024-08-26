FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

WORKDIR /src

RUN apt-get update \
    && apt-get install -y dotnet-sdk-8.0 \
    && apt-get install software-properties-common -y \
    && add-apt-repository ppa:mscore-ubuntu/mscore-stable \
    && apt-get update \
    && apt-get install musescore -y

RUN conda install -c pytorch -c nvidia faiss-gpu=1.8.0

COPY pip-packages.txt pip-packages.txt

RUN pip install --upgrade pip \
    && pip install --upgrade -r pip-packages.txt \
    && rm ./pip-packages.txt

ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root","--no-browser", "--LabApp.token='dev'"]