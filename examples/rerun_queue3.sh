#!/bin/bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4; P=~/venvs/sdos_dev/bin/python; ST=~/sdos_pkg/examples/rerun_status.txt
until grep -q "queue2 end" $ST; do sleep 20; done
cd ~/sdos_pkg/examples/silicon; systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 $P si_time.py 32 48 64 > si_time_clean.log 2>&1; echo "$(date +%H:%M) si_time clean 32,48,64 exit $?" >> $ST
echo "$(date +%H:%M) queue3 end" >> $ST
