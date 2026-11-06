#!/bin/bash
# Part-2 reruns of the sdos revision directive (3 Nov 2026), sequential, 4 threads each, as in the original benchmarks.
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4; P=~/venvs/sdos_dev/bin/python; ST=~/sdos_pkg/examples/rerun_status.txt
note(){ echo "$(date +%H:%M) $*" >> $ST; }
note "queue start"
cd ~/sdos_pkg/examples/silicon; systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 $P si_time.py 32 48 > si_time.log 2>&1; note "si_time 32,48 exit $?"
cd ~/sdos_pkg/examples/bench; systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 $P bench_kwon_k4.py 1 2 3 4 5 6 > bench_kwon_k4.log 2>&1; note "kwon K=4 exit $?"
cd ~/sdos_pkg/examples/bench; systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 $P bench_skf_cheb.py 4 > bench_skf_cheb.log 2>&1; note "dftb chebyshev 262144 exit $?"
cd ~/sdos_pkg/examples/silicon; systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 $P -c "
import sys; sys.argv=['x','64']; exec(open('si_time.py').read().replace('si_time.json','si_time_64.json'))" > si_time_64.log 2>&1; note "si_time 64 exit $?"
note "queue end"
