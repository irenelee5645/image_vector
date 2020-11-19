 python -u main.py \
    --epochs 300 \
    --weight-decay 0.01 \
    --momentum 0.2 \
    --batch-size 128 \
    --num-workers 8 \
    --resume \
    --lr 0.01 | tee logging.log
