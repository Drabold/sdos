#!/bin/bash
# waits for the hot anneal test to finish (GPU), then runs the 17^3 matrix-free ACE vibrational DOS
cd /home/drabold/sdos_pkg/examples/r10_million
while pgrep -f "in.anneal6_hot" >/dev/null; do sleep 15; done
export PATH=/home/drabold/opt/cuda-13.4/bin:$PATH LD_LIBRARY_PATH=/home/drabold/opt/cuda-13.4/lib64:$LD_LIBRARY_PATH OMP_NUM_THREADS=4
/home/drabold/venvs/sdos_dev/bin/python -W ignore r10_vib.py 17 600 2 > r10_vib17_M600.log 2>&1
