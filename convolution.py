import numpy as np 
from scipy import signal
from layer import Layer 


class Convolutional(Layer): 
    """
    For example! 
    if input is grayscale, the shape is: 1 * 28 * 28. 
    with kernel_size 4*4, padding 3, and stride 3, and
    n_filters 5, the output would be 5*11*11. 
    
    thus,
    input_shape  = 1, 28, 28
    input_depth  = 1
    input_height = 28
    input_width  = 28 
    
    output_shape = 5, 11, 11
    kernels_shape= 5, 1, 4, 4
    biases' shape= 5, 11, 11

    """
    def __init__(self, input_shape, kernel_size, n_filters, stride=1, padding=0):
        self.input_shape  = input_shape
        self.input_depth  = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width  = input_shape[2]

        self.n_filters    = n_filters

        self.kernel_size  = kernel_size
        self.stride       = stride 
        self.padding      = padding
        
        # used for calculating shape of other layers
        self.output_height= (self.input_height - self.kernel_size + 2 * self.padding) // self.stride + 1
        self.output_width = (self.input_width  - self.kernel_size + 2 * self.padding) // self.stride + 1

        # stride can not be minus or zero. Must be >0! 
        if self.stride <= 0: 
            raise ValueError("Stride must be greater than zero")

        # raise error to prevent unexpected calculation
        if (self.input_height - self.kernel_size + 2 * self.padding) %  stride != 0 or \
           (self.input_width  - self.kernel_size + 2 * self.padding) %  stride != 0:
            raise ValueError("The stride does not evenly divide output dimensions. Adjust padding, kernel size, and stride")

        self.output_shape = (self.n_filters, self.output_height, self.output_width)

        self.kernels_shape= (self.n_filters, self.input_depth, self.kernel_size, self.kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape) * 0.1
        self.biases  = np.zeros(self.output_shape)
    
    def pad_input(self, input): 
        # use or not use the padding for input data
        if self.padding == 0:
            return input 
        return np.pad(input, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

    def strided_convolution(self, input, kernel, stride):
        # apply correlation and slice output by stride 
        convolved = signal.correlate2d(input, kernel, 'valid')
        return convolved[::stride, ::stride]
           
    def forward(self, input):
        # the sizes below are an example 
        # pad_input: use padding or not?
        # input 28*28, padding=3 -> 34*34 
        self.input = self.pad_input(input) 
        self.output= np.copy(self.biases) 
        for i in range(self.n_filters): 
             for j in range(self.input_depth): 
                self.output[i] += self.strided_convolution(self.input[j], self.kernels[i, j], self.stride)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        # the sizes below are an example! 
        # kernel's gradient shape 5 1 4 4 
        kernels_gradient = np.zeros(self.kernels_shape)
        # input's gradient shape 1 28 28
        input_gradient   = np.zeros(self.input_shape)
        
        dilated_height = (self.output_height - 1) * self.stride + 1
        dilated_width  = (self.output_width  - 1) * self.stride + 1
        
        # dilated's gradient shape 5 31 31
        dilated_gradient = np.zeros((self.n_filters, dilated_height, dilated_width))
        # output's gradient shape 5 11 11
        dilated_gradient[:, ::self.stride, ::self.stride] = output_gradient 

        for i in range(self.n_filters): 
            for j in range(self.input_depth):
                # input j's gradient shape 1 34 34
                
                # 34 - 31 + 1 = 4 -> kernel's shape
                kernels_gradient[i, j] += signal.correlate2d(self.input[j], dilated_gradient[i], 'valid')
                
                # conv's shape 31 31: same with dgr
                convolved = signal.convolve2d(dilated_gradient[i], np.flip(self.kernels[i, j]), 'same')
                
                # check size of convolved & input_gradient
                h, w = input_gradient[j].shape 
                ch, cw  = convolved.shape
                
                start_h = (ch - h) // 2
                start_w = (cw - w) // 2 
                
                # if convolved's shape < input_g's shape
                # for exp: conv 27*27 & input_g's 28*28
                # convolved's structure has to be expanded
                if ch < h or cw < w: 
                    convolved = np.pad(convolved, ((0, h - ch), (0, w - cw)), mode='constant')
                                    
                # if convolved's shape > input_g's shape
                # cut the centered part of the convolve
                # for exp: conv 31*31 & input_g's 28*28
                else: 
                    convolved = convolved[start_h:start_h + h, start_w:start_w + w]
                
                input_gradient[j] += convolved
                     
        # update kernel and bias
        self.kernels -= learning_rate * kernels_gradient
        self.biases  -= learning_rate * np.mean(output_gradient, axis=(1, 2), keepdims=True)
        
        return input_gradient 

# modified from https://github.com/TheIndependentCode/Neural-Network/tree/master
    
    
   
    
    
 






    

    

         

