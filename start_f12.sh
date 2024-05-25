#!/bin/bash
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export DISPLAY=:0
export XAUTHORITY=/home/aounit1/.Xauthority
/home/aounit1/myflaskenv/bin/python /home/aounit1/Desktop/AO/F12.py
