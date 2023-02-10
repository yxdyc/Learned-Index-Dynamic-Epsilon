#!/bin/sh

./DynamicEpsilon_FLOAT --lognormal --pgm --epsilon=16,32,64,128,256 | tee ../results/dynamic_epsilon_lognormal_pgm.txt
./DynamicEpsilon_FLOAT --latilong --pgm --epsilon=16,32,64,128,256 | tee ../results/dynamic_epsilon_map_pgm.txt
./DynamicEpsilon_INT --iot --pgm --epsilon=16,32,64,128,256 | tee ../results/dynamic_epsilon_iot_pgm.txt
./DynamicEpsilon_INT --web --pgm --epsilon=2,4,8,16,32 | tee ../results/dynamic_epsilon_web_pgm.txt
