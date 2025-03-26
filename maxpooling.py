from layer import Layer
import numpy as np 

class MaxPooling2D(Layer):
    def __init__(self, pool_size=2, stride=2): 
        self.pool_size = pool_size 
        self.stride    = stride 
            
    def iterate_regions(self, input):
        # regions: take n*n shape or 2*2 from the input
        n_depth, height, width = input.shape 
        
        # if the input is 5, 11, 11, the output after max
        # pooling would be 5, 5, 5 -> 11//2    
        new_height = (height - self.pool_size) // self.stride + 1
        new_width  = (width  - self.pool_size) // self.stride + 1
          
        # for i in range 11
        for i in range(new_height):
            # for j in range 11
            for j in range(new_width):
                # the i is [0:2] and the j is [0:2]
                # the i = take first and second row
                # the j = take first and second col
                # or 2*2 shape at top left of the array
                image_region = input[:, 
                                     (i * self.stride):(i * self.stride + self.pool_size),
                                     (j * self.stride):(j * self.stride + self.pool_size)]
                yield image_region, i, j 
                
    def forward(self, input):
        # shape of input 5, 11, 11 
        self.last_input = input 
        
        n_depth, height, width = input.shape
        
        new_height = (height - self.pool_size) // self.stride + 1
        new_width  = (width  - self.pool_size) // self.stride + 1
        
        # structure for new input after 2D MaxPooling
        output = np.zeros((n_depth, new_height, new_width))
        
        # take the maximum value of each image_region
        for image_region, i, j in self.iterate_regions(input): 
            output[:, i, j] = np.amax(image_region, axis =(1,2))
        
        # shape of output 5, 5, 5
        return output 
    
    def backward(self, output_gradient, learning_rate): 
        # shape of input 5, 11, 11
        input_gradient = np.zeros(self.last_input.shape)
        
        for image_region, i, j in self.iterate_regions(self.last_input): 
            region_n_depth, region_height, region_width = image_region.shape 
            
            amax = np.amax(image_region, axis = (1, 2)) 
            
            # check each row, col, and find the max value
            for i2 in range(region_height): 
                for j2 in range(region_width):
                    for f2 in range(region_n_depth): 
                        if image_region[f2, i2, j2] == amax[f2]:
                            # gradient is only sent to position of max value (f2) 
                            input_gradient[f2, i*self.pool_size + i2, j*self.pool_size + j2] = output_gradient[f2, i, j]
                                      
        return input_gradient 
    
    # modified from https://github.com/vzhou842/cnn-from-scratch/blob/master/maxpool.py
    
            
        