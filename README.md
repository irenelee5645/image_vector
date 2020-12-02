# Im2Vec
Im2Vec is a language model approach to visualizing and understanding images in image classification task. Im2Vec generates image embedding vectors for each class of image and shows relationship among images of different classes. 

## Setup
Please have Pytorch installed to reproduce our results.
Other libraries required include `numpy` and `pickle` .
In order to visualize the generated vectors, you need to install `matplotlib` as well.


## Dataset
Image embedding vectors can be generated using any dataset used for image classification, but we recommend using small datasets such as CIFAR-10 and CIFAR-100.
In our example, we have used CIFAR-10 and CIFAR-100.

There is no need to download the dataset beforehand, as if the dataset is not found in the specified directory (`data` directory by default),  the dataset will be automatically downloaded. The data directory can be specified with `--cifar100-dir` or `--cifar10-dir` for CIFAR-100 and CIFAR-10 respectively.

## Running the code
The following is an example script to produce Im2Vec with CIFAR-100 dataset.

    python -u main.py \
    	--epochs 1000 \
    	--weight-decay 0.01 \
    	--momentum 0.2 \
    	--batch-size 128 \
    	--num-workers 8 \
    	--classes lion tiger wolf \
    	--resume \
    	--input three_select/lion_tiger_wolf\
    	--output three_select/lion_tiger_wolf2 \
    	--lr 0.01 | tee three_select/ltw.log

### Arguments 
The following are few of the arguments for the main script.
`--classes` : the classes for which the image vector would be generated. All other classes are disregarded and the model is trained only on the selected classes. (This argument only exists for `main.py` which is designed for CIFAR-100 specifically. `cifar10_main.py`, which is designed for CIFAR-10 trains the model on all existing classes.
`--resume` : loads the `pt` file specified by `--input` and resumes training from there.
`--input`: the `pt` file that has weights from which the training will be resumed
`--output` : the output `pt` file to store trained model.

## Generating Image Embedding Vectors
Run the Jupyter notebook in `threeselect/` to generate image embedding vectors for selected classes from CIFAR-100 and that in `cifar10/` to generate the vectors for CIFAR-10 classes.

## References
Word2Vec : https://arxiv.org/pdf/1301.3781.pdf
The code for the main script has been adapted from Georgia Tech's CS 4803 assignment.
