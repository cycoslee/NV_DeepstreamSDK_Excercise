#!/bin/bash
apt-get update
apt-get install -y python3-pip
python3 -m pip install Cython
python3 -m pip install numpy
python3 -m pip install pandas
apt-get install -y python3-matplotlib
apt-get install -y python3-cairocffi
