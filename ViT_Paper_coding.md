WHAT IS PAPER REPLICATING?
 The goal of paper replicating is to replicate advance machine learning research papers with code so we can use the techniques for our own problems.
 For example, let's say that a new model architecture gets released that performs better than any other architecture before on various benchmarks, wouldn't it be nice to try that architecture on our own problems.

Machine learning paper replicating involves turning a machine learning paper comprised of images/diagrams, math and text into usable code.

 Code examples for machine learning research papers?

 RESOURCE: 1. arXiv
           2.  Papers with code

# We are going to replicate the machine learning research paper "An image is worth 16x16 words, Transformers for image recognoition at scale"(ViT paper) with PyTorch

A Transformer architecture is generally considered to be neural network that uses the attention mechanism as its primary learning layer.Similar to convolutional neural network.
ViT(Vision Transformer) Architecture was designed to adapt the origianl transformer architecture to vision problems(classification being the first and others followed)



# REPLICATING THE ViT PAPER:
  replicate ViT paper for our foodvision mini problem.
  our model inputs are : pizza, steak, sushi.
  our ideal model outputs are: predicted labels of pizza, steak, sushi.

  # Terminology:
  1. Layer: takes an input, performs function on it, returns an output.
  2. Block: a collection of layers, takes an input, performs a series of fucntions on it and gives an output.
  3. Architecture: A collection of blocks, takes an output, performs a series of functions on it, returns an output.

  Go through ViT paper layer by layer, block by block, function by function putting the pieces together.


# The ViT architecture is composed of several stages:
1. Patch + Postional Embedding(Inputs) : Turns the input image into a sequence of image patches and adds a position number to specify in what order the patches come in.
2. Linear Projection of Flattened patches(Embedded patches) : The image patches get turned into a embedding, the benefit of using an embedding rather than just the image values is that an embedding
   is a learnable representation(typically in the form of a vector) of the image that can improve with training.
3. Norm : short for "Layer Normalization" or "LayerNorm", a technique for regularizing(reducing overfitting) a neurral network,  used via "torch.nn.LayerNorm()".
4. Multi-Head Attention: "Multi-Headed self-Attention layer" or "MSA", created via "torch.nn.MultiheadAttention()".
5. MLP(MultiLAyer Perceptron) : A MLP can refer to any collection of feed forward layers(PyTorch case --colllection of layres of forward() method).
   In ViT paper, the MLP is referred to as "MLP Block" and it contains two "torch.nn.Linear()" layers with a "torch.nn.GELU()" non-linearity activation in between them and a "torch.nn.Dropout()"
   layer after each.
6. Transformer Encoder : The transformer Encoder, is a collection of layers listed above. There are two skip conenctions inside the Transformer encoder(the "+" symbols) meaning the layers inputs
   fed directly to immediate layers as well as subsequent layers. The overall ViT architecture is comprised of a number of Transformer encoders stacked on top of each other.
7. MLP Head : This is the output layer of the architecture, it converts the learned features of an input to a class output.Since we're working on image classification, we can call this the\
   "classifier head". The structure of MLP head is similar to the MLP block.


# THE FOUR EQUATIONS OF THE ViT PAPER IN SECTION 3.1:
# OVERVIEW :
1. EQUATION 1 OVERVIEW:

 
   The transformer uses constant latent vector size D through all of its layers, so we flatten the patches and map D dimensions with a trainable linear projection.
   We refer to the  output of this projection as "patch embedding".
   "Position Embeddings are added to the patch embeddings to retain positional information. We use standard learnable 1D positional embeddings.

   The equation deals with the class token, patch embedding anf position embedding(E for embedding) of the input image.

   in vector form,
   
   x_input = [class_token, image_patch1, image_patch2, image_patch3...] + [class_token_position, image_patch1_position, image_patch2_position, image_patch3_position...]
   where each of the elements in the vector is learnable (their requires_grad=True)
   

2. EQUATION 2 OVERVIEW:

   The transformer encoder consists of altenating layers of multiheaded selfattention(MSA) and MLP blocks. LayerNorm(LN) is applied before every block and residual connections after every block.

   The equation says that for every layer from 1 through to L(total number of layers), there is a Multi-headed Self-Attention layer(MSA) wrapping a LayerNorm layer(LN).
   The addition at the end is the equivalent of adding the input to the output and forming a skip/residual connection.
   Pseudocode:

   x_output_MSA_block = MSA_layer(LN_layer(x_input)) + x_input

3. EQUATION 3 OVERVIEW:

   The transformer encoder consists of altenating layers of multiheaded selfattention(MSA) and MLP blocks. LayerNorm(LN) is applied before every block and residual connections after every block.

   The equation says that for every layer from 1 through to L(total number of layers), there is a Multilayer Perceptron layer(MLP) wrapping a LayerNorm layer(LN).
   The addition at the end is the equivalent of adding the input to the output and forming a skip/residual connection.
   
   Pseudocode:

   x_output_MLP_block = MLP_layer(LN_layer(x__output_MSA_block)) + x_output_MSA_block

4. EQUATION 4 OVERVIEW:

   Similar to BERT's[class] token, we prepared a learnble embedding to the sequenceof embedded patches(z=x(class)) whose state at the output of the Transformer encoder serves as the imaginary representation y.

   the equation says that for the last layer L, the output y is the 0 index token of z wrapped in a LayerNorm(LN).
   or, the 0 index of x_output_MLP_block:

   y = Linear_layer(LN_layer(x_output_MLP_block[0]))


# Focus on the Table 1 from the Vit paper showcasing the various hyperparameters of each of the ViT architecture:

Hyperparameters:
1. Layers: How many transformer encoder blocks are there? (Each of these will contain a MSA block and MLP block)
2. Hidden size D : This is the embedding dimension throughout the architecture, this will be the size of the vector that our image gets turned into when it gets patched and embedded.
   Generally, the larger the embeddding dimension, the more information can be captured, better the results. However, a larger embedding comes at the cost of more computation.
3. MLP size : What are the number of hidden units in the MLP layers?
4. Heads : How many heads are there in the Multi-headed Attention layers?
5. Params : What are the total number of parameters of the model?
   Generally, more parameters leads to better performance but at the cost of more computation.

     

   

















   
     
  