class Layer: 
    def __init__(self): 
        self.input = None 
        self.output= None 

    def forward(self, input): 
        pass 

    def backward(self, output_gradient, learning_rate): 
        pass 
    
# taken from https://github.com/TheIndependentCode/Neural-Network/tree/master