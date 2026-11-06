#!/bin/bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4; P=~/venvs/sdos_dev/bin/python; ST=~/sdos_pkg/examples/rerun_status.txt
note(){ echo "$(date +%H:%M) $*" >> $ST; }
until grep -q "queue end" $ST; do sleep 20; done
cd ~/sdos_pkg/examples/bench; systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 $P bench_kwon_k4.py 1 2 3 4 5 6 > bench_kwon_k4.log 2>&1; note "kwon K=4 (fixed exact.py) exit $?"
cd ~/sdos_pkg; systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 $P -m pytest -x -q > /tmp/claude-1000/-home-drabold-lammps-runs/c940f31d-33c2-4bcf-91f9-093319989923/scratchpad/pytest_after_exactfix.out 2>&1; note "pytest exit $?"
note "queue2 end"
