#!/bin/sh

# test naive varied epsilon
#dataset=iot
dataset=toy_breaking
init_epsilon=128
min_epsilon=16
max_epsilon=512
add_rate=0.25
mul_rate=2
adjust_manner=MUL
decay_threshold=0.2
increase_threshold=0.05
w_rate=0.99
ablate_meta_learner=
small_batch_size=-1
look_ahead_n=500
./cmake-build-relwithdebinfo/VariedEpsilon --$dataset --fitting-tree --w_rate $w_rate --init_epsilon $init_epsilon --min_epsilon $min_epsilon --max_epsilon $max_epsilon --mul_rate $mul_rate --adjust_manner $adjust_manner --decay_threshold $decay_threshold --increase_threshold $increase_threshold $ablate_meta_learner --small_batch_size $small_batch_size --look_ahead_n $look_ahead_n | tee ./results/varied_epsilon/iot_fitting_${w_rate}/${adjust_manner}_${add_rate}_${init_epsilon}.txt


