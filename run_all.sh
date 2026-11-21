#!/bin/sh
# Regenerates every result file and figure. Takes roughly ten minutes on a laptop.
set -e
mkdir -p figs
python run_model.py   > /dev/null    # results.json
python kernel.py      > /dev/null    # kernel.json
python entropy.py     > /dev/null    # entropy.json
python joint_entropy.py > /dev/null  # joint.json
python selfent.py     > /dev/null    # selfent.json
python robust.py      > /dev/null    # robust.json
python relevance.py 0.8 > /dev/null  # relevance_T0.8.json
python ib.py          > /dev/null    # ib.json
python sim2d.py       > /dev/null    # sim2d.json
python kawasaki.py 1 1  > /dev/null  # kawasaki_lag1_seed1.json   (4 sweeps ahead)
python kawasaki.py 8 1  > /dev/null  # kawasaki_lag8_seed1.json   (32 sweeps ahead)
for seed in 1 2 3; do python kawasaki.py 32 $seed > /dev/null; done   # 128 sweeps ahead
python exact_check.py
python make_figures.py               # figs/*.pdf
echo done
