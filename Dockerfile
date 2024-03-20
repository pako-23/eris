FROM python:3.9-slim-bookworm AS builder

WORKDIR /app

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
    cmake \
    g++ \
    git

COPY . .
RUN pip install -r requirements.txt
RUN ./setup.py bdist_wheel