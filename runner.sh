

 python -u main.py \
    --epochs 700 \
    --weight-decay 0.01 \
    --momentum 0.2 \
    --batch-size 128 \
    --num-workers 8 \
    --classes lion tiger wolf \
    --resume \
    --input three_select/lion_tiger_wolf\
    --out three_select/lion_tiger_wolf2 \
    --lr 0.01 | tee three_select/ltw.log

 python -u main.py \
    --epochs 700 \
    --weight-decay 0.01 \
    --momentum 0.2 \
    --batch-size 512 \
    --num-workers 8 \
    --classes tulip rose sunflower \
    --resume \
    --input three_select/tulip_rose_sunflower \
    --output three_select/tulip_rose_sunflower2 \
    --lr 0.01 | tee three_select/trs2.log


 python -u main.py \
    --epochs 1000 \
    --weight-decay 0.01 \
    --momentum 0.2 \
    --batch-size 512 \
    --num-workers 8 \
    --classes bicycle bus motorcycle train \
    --out three_select/vehicles1 \
    --lr 0.01 | tee three_select/vehicles1.log


 python -u main.py \
    --epochs 1000 \
    --weight-decay 0.01 \
    --momentum 0.2 \
    --batch-size 512 \
    --num-workers 8 \
    --classes baby boy girl man woman \
    --out three_select/people \
    --lr 0.01 | tee three_select/people.log


 python -u main.py \
    --epochs 1000 \
    --weight-decay 0.01 \
    --momentum 0.2 \
    --batch-size 512 \
    --num-workers 8 \
    --classes camel cattle chimpanzee elephant kangaroo \
    --out three_select/largeob \
    --lr 0.01 | tee three_select/largeob.log

 python -u cifar10_main.py \
    --epochs 500 \
    --weight-decay 0.01 \
    --momentum 0.2 \
    --batch-size 512 \
    --out cifar10/cifar10 \
    --lr 0.01 | tee cifar10/cifar10.log