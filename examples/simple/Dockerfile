FROM eris/eris AS builder

FROM builder AS coordinator

COPY coordinator.py /app/coordinator.py

CMD ["./coordinator.py"]

FROM builder AS client

COPY model.py /app/model.py
COPY client.py /app/client.py

CMD ["./client.py"]