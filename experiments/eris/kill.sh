#!/bin/sh -eu


while pgrep -f ./client.py >/dev/null; do
    sleep 5
done

pkill -9 -f coordinator.py
pkill -9 -f client.py
