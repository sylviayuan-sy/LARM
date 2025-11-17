export MASTER_ADDR=127.0.0.1
export MASTER_PORT=6006
export JOB_COMPLETION_INDEX=0
NCCL_DEBUG=INFO torchrun --nproc_per_node 2 --nnodes 1 \
    --master-port ${MASTER_PORT} \
    --master-addr ${MASTER_ADDR} \
    --node-rank ${JOB_COMPLETION_INDEX} \
    trainer_mask.py \
    --config=configs/part.yaml 