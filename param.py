#import archsSEMlp
#import torch
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#ARCH_NAMES = archsSEMlp.__all__
#model = archsSEMlp.__dict__['UNext'](1,3,False)
#model = model.to(device)
#model = model.load_state_dict(torch.load('model.pth', map_location='cpu'))
#model = 你自己的模型
#print('wada', model.parameters())
#params = list(model.parameters())
#k = 0
#for i in params:
#    l = 1
#    print("该层的结构：" + str(list(i.size())))
#    for j in i.size():
#        l *= j
#    print("该层参数和：" + str(l))
#    k = k + l
#print("总参数数量和：" + str(k))


from archs import  UNet
import torch
import utils
from thop import profile
import archsSEMlp
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ARCH_NAMES = archsSEMlp.__all__
model = archsSEMlp.__dict__['UNext'](1,3,False)
model = model.to(device)
model1 = UNet(n_channels=3, n_classes=1, bilinear=False)
model1 = model1.to(device)
#model = model.load_state_dict(torch.load('model.pth', map_location='cpu'))
print(utils.count_params(model))
print(utils.count_params(model1))
#input = torch.randn(1, 3, 256, 256)
#flops, params = profile(model, inputs=(input, ))
#print('flops:{}'.format(flops))
#print('params:{}'.format(params))