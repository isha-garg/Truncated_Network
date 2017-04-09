#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=isha_try/cifar10_truncated/trunc_cifar10_full_solver.prototxt --weights=examples/cifar10/cifar10_full_iter_60000.caffemodel.h5 $@

