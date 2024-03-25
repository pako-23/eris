FROM eris AS coordinator

COPY coordinator.py /app/coordinator.py

CMD ["./coordinator.py"]

FROM eris AS client

COPY model.py /app/model.py
COPY client.py /app/client.py

CMD ["./client.py"]