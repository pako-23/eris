#!/bin/sh -eu


while pgrep -f ./client.py >/dev/null; do
    sleep 5
done

pkill -u dario -f client.py
pkill -u dario -f coordinator.py
pkill -9 -f coordinator.py
pkill -9 -f client.py
