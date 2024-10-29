The project is simply a replication of ViT paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale".
The data set used is borrowed from https://github.com/mrdbourke/pytorch-deep-learning/tree/main/data.
Instead of using the batch size of 4096 as done in the paper, due to hardware constraints we are using a batch size of 32.
