FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

ENV CC=mpicc
ENV CXX=mpicxx
ENV MPICC=mpicc
ENV MPICXX=mpicxx

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        pkg-config \
        openmpi-bin \
        libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY main.py app.py ./
COPY src ./src
COPY dashboard ./dashboard
COPY specifics ./specifics
COPY tests ./tests

RUN which mpicc \
    && which mpicxx \
    && mpicc --version \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install mpi4py \
    && python -m pip install -e ".[dev]"

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]