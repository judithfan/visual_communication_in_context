#!/bin/sh
for i in 1 2 3 4 5
do
  cd raw/balancedavg${i}
  head -1 balancedavg${i}_sim_1_enumerate.csv > ../../balancedavg${i}.csv;
  tail -n +2 -q balancedavg${i}_sim_* >> ../../balancedavg${i}.csv;
  cd ../..
done
