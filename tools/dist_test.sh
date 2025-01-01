#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
# PORT=${PORT:-29835}
# PORT=${PORT:-29836}
PORT=${PORT:-29837}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
