#!/bin/sh -eu

./coordinator.py &
sleep 1.5

for i in $(seq 1 5); do
    ./client.py "$(expr "50051" + "$i")" "$(expr "5555" + "$i")" &
    sleep 0.2
done
for i in $(seq 1 5); do
    ./client.py &
done


while pgrep -f ./client.py >/dev/null; do
    sleep 5
done

pkill -9 -f coordinator.py
