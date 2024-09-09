#自行编写的通用MLP类
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:  # Add activation only between layers, not after output
                layers.append(nn.ReLU())
                
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Define a function to create the model with the desired layers
def common_MLP():
    input_size = 3  # Corresponding to (x, y, t)
    hidden_sizes = [6,6]  # Corresponding to the structure in the original KAN model
    output_size = 3  # Output size for (u, v, p)
    return MLP(input_size, hidden_sizes, output_size)

def MLP_for_NS():
    model = nn.Sequential(
        nn.Linear(2, 6),  # Assuming input features are 2 (x and y coordinates)
        nn.ReLU(),
        nn.Linear(6, 6),
        nn.ReLU(),
        nn.Linear(6, 3)    # Assuming you have 3 outputs (u, v, p)

#        nn.Linear(2, 50),  # Assuming input features are 2 (x and y coordinates)
#        nn.ReLU(),
#        nn.Linear(50, 50),
#        nn.ReLU(),
#        nn.Linear(50, 3)    # Assuming you have 3 outputs (u, v, p)
    )
    return model

def MLP_for_PE():
    input_size = 2  # 输入的特征为2（x 和 y 坐标）
    hidden_sizes = [360,360]  # 隐藏层结构
    output_size = 1  # 预测标量场phi
    return MLP(input_size, hidden_sizes, output_size)

    
    
if __name__ == "__main__":
    # Quick test to ensure the model works
    model = create_mlp_model()
    test_input = torch.randn((5, input_size))  # batch of 5 samples
    test_output = model(test_input)
    print(test_output)
import torch.nn as nn


