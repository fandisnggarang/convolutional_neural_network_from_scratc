# Convolutional Neural Network from scratch! 

I started learning about neural networks (NN) in November 2024, then continued with convolutional neural networks (CNN) in February 2025. I had to pause my studies in January and February due to work and personal matters. By the end of March, I had completed a simple construction of NN and CNN by learning how to build them from scratch. Just like when I studied more traditional machine learning algorithms (which I previously posted in other repo), in this phase of learning NN and CNN, I "observed" code from various sources and tried modifying it.  

In short, the most challenging part of understanding both is the backpropagation process and its mathematics—how the learning process is reevaluated when predictions do not match "reality". The hardest part is understanding the chain rule in calculating derivatives. Complicated? In other words, predictions are generated from chains of information (forward), and when the predictions are incorrect, a backward process must be performed to adjust the parameters at each information chain.  

However, understanding derivatives and the chain rule is not enough, as this process can also lead to numerical instability, causing gradient calculations to become extremely large or small (exploding or vanishing gradients). Simply put, the complexity of both the prediction (forward) and evaluation (backward) processes can become so intricate that the model ends up learning very poorly or not learning at all. 

Another equally exhausting challenge is understanding dimensions or shapes. Both NN and CNN involve layers that perform matrix operations. Many times, the code I wrote resulted in matrices with mismatched dimensions, leading to errors when performing multiplication or convolution. Visualizing the dimensions or shapes in forward propagation is relatively easy. The real challenge is when trying to picture them during the "reverse journey" of backpropagation. "Wait! Why does this need to be transposed?" is a thought that often pops into my mind.  
 
As I mentioned, the following convolutional neural network from scratch is a modified version of code originally created by TheIndependentCode and vzhou842 (Victor Zhou). The modifications I made include:  
1. Adding stride and padding strategies so that the method is not limited to just forward and backward propagation.  
2. Adding a `MaxPooling2D` class that simplifies the input matrix into a two-dimensional size.  
3. Rewriting the code for mean squared error and binary cross-entropy, along with their derivatives.  
4. Separating the model testing process into two distinct methods: `train` and `test`, using the MNIST dataset.  

What I like about the layer structure presented by TheIndependentCode is that it uses the concept of "inheritance", linking one class to another. This helped me understand the concept more intuitively.  

Enjoy!  

