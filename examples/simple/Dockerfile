FROM eris/eris AS eris

FROM pytorch/pytorch:latest AS builder

WORKDIR /app

COPY --from=eris /app/dist/eris-0.0.1-cp312-cp312-linux_x86_64.whl .
RUN pip install eris-0.0.1-cp312-cp312-linux_x86_64.whl \
    && rm eris-0.0.1-cp312-cp312-linux_x86_64.whl

FROM builder AS coordinator

COPY coordinator.py /app/coordinator.py

CMD ["./coordinator.py"]

FROM builder AS client

COPY model.py /app/model.py
COPY client.py /app/client.py

CMD ["./client.py"]