#  python -u main.py \
#     --epochs 300 \
#     --weight-decay 0.01 \
#     --momentum 0.2 \
#     --batch-size 128 \
#     --num-workers 8 \
#     --resume \
#     --output out2 \
#     --input out \
#     --lr 0.01 | tee logging2.log

   #  --resume \

python -u main.py \
    --epochs 300 \
    --weight-decay 0.01 \
    --momentum 0.2 \
    --batch-size 128 \
    --num-workers 8 \
    --output depth2_sig \
    --model dnn \
    --depth 2 \
    --act sig \
    --lr 0.01 | tee depth2_sig.log

 python -u main.py \
    --epochs 300 \
    --weight-decay 0.01 \
    --momentum 0.2 \
    --batch-size 128 \
    --num-workers 8 \
    --classes lion tiger wolf \
    --out three_select/lion_tiger_wolf \
    --lr 0.01 | tee three_select/ltw.log