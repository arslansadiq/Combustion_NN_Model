import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super(MnistModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
class CombustionModel(BaseModel):
    def __init__(self, num_features=7):
        super(CombustionModel, self).__init__(num_features)
        self.Fc1 = nn.Linear(in_features = 2, out_features = 500, bias=True)
        self.Fc2 = nn.Linear(in_features = 500, out_features = 500, bias=True)
        self.Fc3 = nn.Linear(in_features = 500, out_features = 500, bias=True)
        self.Fc4 = nn.Linear(in_features = 500, out_features = 500, bias=True)
        self.Fc5 = nn.Linear(in_features = 500, out_features = 500, bias=True)
        self.Fc6 = nn.Linear(in_features = 500, out_features = 500, bias=True)
        self.Fc7 = nn.Linear(in_features = 500, out_features = 500, bias=True)
        self.Fc8 = nn.Linear(in_features = 500, out_features = 500, bias=True)
        self.Fc9 = nn.Linear(in_features = 500, out_features = 500, bias=True)
        self.Fc10 = nn.Linear(in_features = 500, out_features = 500, bias=True)
        self.Fc11 = nn.Linear(in_features = 500, out_features = 500, bias=True)
        self.Fc12 = nn.Linear(in_features = 500, out_features = num_features, bias=True)
    
    def forward(self, x):
        '''
        This function computes the network computations based on input x 
        built in the constructor of the the CombustionModel
        '''
        
        '''First Layer'''
        x = self.Fc1(x)
        x = F.relu(x)
        
        '''First ResNet Block'''
        res_calc = self.Fc2(x)
        res_calc = F.relu(res_calc)
        res_calc = self.Fc3(res_calc)
        x = F.relu(torch.add(x, res_calc))
        
        '''Second ResNet Block'''
        res_calc = self.Fc4(x)
        res_calc = F.relu(res_calc)
        res_calc = self.Fc5(res_calc)
        x = F.relu(torch.add(x, res_calc))
        
        '''Third ResNet Block'''
        res_calc = self.Fc6(x)
        res_calc = F.relu(res_calc)
        res_calc = self.Fc7(res_calc)
        x = F.relu(torch.add(x, res_calc))
        
        '''Fourth ResNet Block'''
        res_calc = self.Fc8(x)
        res_calc = F.relu(res_calc)
        res_calc = self.Fc9(res_calc)
        x = F.relu(torch.add(x, res_calc))
        
        '''Fifth ResNet Block'''
        res_calc = self.Fc10(x)
        res_calc = F.relu(res_calc)
        res_calc = self.Fc11(res_calc)
        x = F.relu(torch.add(x, res_calc))
        
        '''Regression layer'''
        return self.Fc12(x)
        
