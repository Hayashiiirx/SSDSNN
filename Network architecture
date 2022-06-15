import torch
from torchvision import datasets, transforms
from torch import _pin_memory, nn  
from spikingjelly.clock_driven import neuron, encoding, functional, surrogate, layer
from tqdm import tqdm
from torch.nn import functional as F
import time
import math
from torch.nn import init
from typing import Callable
import numpy as np
#from torch.utils.tensorboard import SummaryWriter
import argparse


parser = argparse.ArgumentParser(description='Classify Fashion-MNIST')
parser.add_argument('-T', default = 16, type=int, help='simulating time-steps')
parser.add_argument('-device', default='cuda:4', help='device')
parser.add_argument('-batch_size', default=32, type=int, help='batch size')
parser.add_argument('-epochs', default=200, type=int, metavar='N',help='number of total epochs to run')
parser.add_argument('-num_worker', default=8, type=int, metavar='N',help='number of data loading workers (default: 4)')
parser.add_argument('-data_dir', default = '/home/lrx/SNN代码/data/', type=str, help='root dir of dataset')
parser.add_argument('-out_dir', type=str, default='log-dir/', help='root dir for saving logs and checkpoint')
parser.add_argument('-opt', default = 'SGD', type=str, help='use which optimizer. SGD or Adam')
parser.add_argument('-lr', default=0.01, type=float, help='learning rate')
parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
parser.add_argument('-lr_scheduler', default='CosALR', type=str, help='use which schedule. StepLR or CosALR')
parser.add_argument('-T_max', default=64, type=int, help='T_max for CosineAnnealingLR')
parser.add_argument('-V', default=1.0, type=float, help='Threshold of spiking neuron')
parser.add_argument('-eta_min', default= 0.0001, type = float, help = 'The minimum value of the learning rate to the CosALR')
parser.add_argument('-dataset', type = str, default= 'mnist', help = 'choose Fashionmnist or mnist')
args = parser.parse_args()



"""
_seed_ = 2022
torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)
"""

def Sigma_Delta_Encoder(X : torch.Tensor, t : type = int, dim : type = int, 
                        V_t : float = 0, pooling : bool = False, pooling_num : int = 0,
                        pooling_stride : int = 0, mode : str = 'avg'):
    """
    * : 使用一阶单比特``Sigma-Delta ADC``对输入图像进行编码。该函数的输出为一个长度为T的列表，
        列表每个元素都是对应时刻的编码内容。
    
    :X : 输入的频谱图或者图片四维张量矩阵；
    
    : t : 所执行的时间步数目；
    
    :dim : 对输入张量矩阵按照不同维度进行编码，只能为0或1，
            对于频谱图，该项为0时，即为按照时间维度进行编码；
            
    : V_t : 发放脉冲的阈值；
    
    : pooling :对于输入的不同大小的张量矩阵，使用池化可以使之尺寸相同；
    
    : pooling_num : 如果``pooling``为``True``，则给定池化大小，注意此处池化为一维池化；
    """
    
    # 不使用池化
    if not pooling and dim == 0: 
                    out = []
                    feedback = (-1 * torch.ones(X.shape[0], X.shape[1], X.shape[2])).to(args.device)
                    integrator = torch.zeros(X.shape[0], X.shape[1], X.shape[2]).to(args.device)
                    for time_step in range(t):
                        d = torch.zeros(X.shape).to(args.device)
                        for j in range(X.shape[-1]):
                            #print(X[ : , : , : , j].shape)
                            V_m = X[ : , : , : , j] - feedback + integrator # 比较器的输入
                            d[: , : , : ,j] = ((V_m - V_t) > 0) * torch.ones(feedback.shape).to(args.device)    # 量化编码
                            feedback = ((V_m - V_t) > 0) * 2 - 1    # 输出0返回1，输出1返回-1
                            integrator = V_m
                            #print('feedback_epoch:', feedback)
                            #print('integator_epoch:', integrator)
                        out.append(d)
                            
    elif not pooling and dim == 1:
                out = []
                feedback = (-1 * torch.ones(X.shape[0], X.shape[1], X.shape[3])).to(args.device)
                integrator = torch.zeros(X.shape[0], X.shape[1], X.shape[3]).to(args.device)
                for time_step in range(t):
                        d = torch.zeros(X.shape).to(args.device)
                        for j in range(X.shape[2]):
                            V_m = X[ : , : , j , :] - feedback + integrator # 比较器的输入
                            d[: , : , j, : ] = ((V_m - V_t) > 0) * torch.ones(feedback.shape).to(args.device)    # 量化编码
                            feedback = ((V_m - V_t) > 0) * 2 - 1    # 输出0返回1，输出1返回-1
                            integrator = V_m
                        out.append(d)
    else:
        out = []
    return out
    """
        elif pooling:
            if dim == 0: 
                    Iter = torch.zeros(X.shape[0], X.shape[1], X.shape[2], X.shape[3] - pooling_num + 1)
                    for i in range(Iter.shape[3]):
                            if mode == 'max':
                                Iter[:, :, :, i] = X[: , : , : , i  :  : , : , : , i + pooling_num].max()
                            elif mode == 'avg':
                                Iter[:, :, :, i] = X[: , : , : , i, :  : , : , : , i + pooling_num].mean()
                    feedback = -1 * torch.ones(Iter.shape[0], Iter.shape[1], 1, Iter.shape[3])
                    integrator = torch.zeros(Iter.shape[0], Iter.shape[1], 1, Iter.shape[3])
                    for time_step in range(t):
                            for j in range(Iter.shape[2]):
                                V_m = Iter[ : , : , j , :] - feedback + integrator # 比较器的输入
                                d[: , : , : ,j] = ((V_m - V_t) > 0) * torch.ones(feedback.shape)    # 量化编码
                                feedback = ((V_m - V_t) < 0) * 2 - 1    # 输出0返回1，输出1返回-1
                                integrator += V_m
                                
            else:
                    Iter = torch.zeros(X.shape[0], X.shape[1], X.shape[2] - pooling_num + 1, X.shape[3] )
                    for i in range(Iter.shape[2]):
                        for k in range(Iter.shape[0]):
                            for j in range(Iter.shape[1]):
                                if mode == 'max':
                                    Iter[k, j, j, :] = X[ : , : , j , :  :  : , : , j + pooling_num , : ].max()
                                elif mode == 'avg':
                                    Iter[:, :, :, j] = X[: , : , : , j, :  : , : , : , j + pooling_num].mean()
                    feedback = -1 * torch.ones(Iter.shape[0], Iter.shape[1], 1, Iter.shape[3])
                    integrator = torch.zeros(Iter.shape[0], Iter.shape[1], 1, Iter.shape[3])
                    for time_step in range(t):
                            for j in range(Iter.shape[2]):
                                V_m = Iter[ : , : , j , :] - feedback + integrator # 比较器的输入
                                d[: , : , : ,j] = ((V_m - V_t) > 0) * torch.ones(feedback.shape)    # 量化编码
                                feedback = ((V_m - V_t) < 0) * 2 - 1    # 输出0返回1，输出1返回-1
                                integrator += V_m
            """

# 将输入值归一化到-1至1的范围内
def Dual_mormalization(X : torch.Tensor, mapping : str = 'linear'):
    if mapping == 'linear':
        X = 2 * X - 1
    elif mapping == 'Sinc':
        X = torch.sin(X * math.pi  - math.pi * 0.5)
    return X


# 扩充反向传播维度(一维-->二维)
def multi_cat(x : torch.Tensor, n, dim = 0):
    a = x
    for i in range(n-1):
        x = torch.cat((a, x), dim)
    return x

# 扩充反向传播维度(二维-->四维)
def Conv_multi_cat(x : torch.Tensor, n, ):
    b = (x[1].clone()).reshape(x.shape[1], 1)
    x = torch.unsqueeze(x, 2)
    x = torch.unsqueeze(x, 3)
    
    for j in range(x.shape[1]):
        if x.shape[2] != n and x.shape[3] != n:
            x = multi_cat(x, n, dim = 2)
            x = multi_cat(x, n, dim = 3)
        else:
            x[ :, j, :, :] = b[j]
    return x


# 根据spikingjelly更改激活函数
class srelu(torch.autograd.Function):
    # 手动定义前项公式
    @staticmethod
    def forward(ctx, x, bias, v_th):
        # x：[128, 6, 28, 28], [128, 16, 10, 10], [128, 120], [128, 84]
        # bias: [6], [16], [120], [84], 
        #print('spikebias:',bias)
        ctx.bias0 = bias
        if x.requires_grad:
            ctx.save_for_backward(x)# 将输入保存起来，在backward时使用
            ctx.bias =  bias
            ctx.v_th = v_th
        return surrogate.heaviside(x)
    
    # 手动定义反向传播
    @staticmethod
    def backward(ctx, grad_output):
        
        input_= ctx.saved_tensors[0]  # 得到x
        grad_input = grad_output.clone()
        if ctx.needs_input_grad[0]:
            if ctx.bias0 != None:
                ctx.bias = ((ctx.v_th - ctx.bias) > 0) * ctx.bias + 0.9 * ((ctx.bias - ctx.v_th) >= 0)
                ctx.bias = multi_cat(ctx.bias.reshape(1, ctx.bias.shape[0]), grad_output.shape[0])
                if grad_input.shape == ctx.bias.shape:
                    grad_input = (grad_input * (input_ > 0)) * (1 / (ctx.v_th - ctx.bias))
                else:
                    ctx.bias = ((1 - ctx.bias) >= 0) * ctx.bias + 0.9 * ((ctx.bias - 1) > 0)
                    n = grad_input.shape[2]
                    ctx.bias = Conv_multi_cat(ctx.bias, n)
                    grad_input = (grad_input * (input_ > 0)) * (1 / (ctx.v_th - ctx.bias))
            else:
                grad_input = (grad_input * (input_ > 0)) * (1 / (ctx.v_th))
            ctx.bias = ctx.bias0
        return grad_input, None, None
    
class SRelu(surrogate.SurrogateFunctionBase):
    def __init__(self, alpha=4.0, spiking=True, bias = 0 , v_th = 1.0):
        
        super().__init__(alpha=4.0, spiking=True)
        self.bias, self.v_th = bias, v_th
        self.spiking = spiking
    @staticmethod
    def spiking_function(x, bias, v_th ):
        return srelu.apply(x, bias, v_th )
    
    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return (x * alpha).srelu()
    
    def forward(self, x: torch.Tensor):
        if self.spiking:
            return self.spiking_function(x, self.bias, self.v_th)
        
# 融合neuron.IFNode与nn.Linear
class IFNeuron(neuron.IFNode):
    def __init__(self, input_size, output_size, bias : bool = True ,v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = SRelu(), detach_reset: bool = False):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        self.v_threshold = v_threshold
        
        self.weight = nn.Parameter(torch.empty((output_size, input_size)))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
 
        if type(surrogate_function) == SRelu:
            #print('self.bias:', self.bias)
            self.surrogate_function = SRelu(bias = self.bias)
        else:
            #print(type(surrogate_function))
            self.surrogate_function = surrogate_function
    
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            #print('bound',bound)
            init.uniform_(self.bias, -bound, bound)
            
    def neuronal_charge(self, x : torch.Tensor):
        #print('膜电位增加值：',torch.matmul(x, self.weight.data) + self.bias.data)
        self.v = self.v + F.linear(x, self.weight, self.bias)
        #print('膜电位：', self.v)
        
    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold)
    
    def forward(self, x: torch.Tensor):
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike
    
    
# neuron.IFNode与nn.Conv2d
class Conv2d_IFNeuron(neuron.IFNode):
    def __init__(self, input_channels, output_channels, kernel_size, stride = 1, padding = 0
                 ,bias : bool = True, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = SRelu(), detach_reset: bool = False):
        neuron.IFNode.__init__(self, v_threshold, v_reset, surrogate_function, detach_reset)
        self.v_threshold = v_threshold
        self.in_channels = input_channels
        self.out_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.Conv2d = nn.Conv2d(in_channels = self.in_channels, out_channels= self.out_channels, 
                               kernel_size= self.kernel_size, stride= self.stride, padding = self.padding, bias = bias)
        self.bias = self.Conv2d.bias
        self.weight = self.Conv2d.weight
        
        if type(surrogate_function) == SRelu:
            #print('self.bias:', self.bias)
            self.surrogate_function = SRelu(bias = self.bias)
        else:
            #print(type(surrogate_function))
            self.surrogate_function = surrogate_function
            
    def neuronal_charge(self, X : torch.Tensor):
        self.v = self.v + self.Conv2d(X)
    
    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold)
     
    def forward(self, X: torch.Tensor):
        self.neuronal_charge(X)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike
    
# 没有编码器，使用第一层为编码层，并且不参与时间循环中
class PythonNet(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

        self.static_conv = nn.Sequential(
                                        Conv2d_IFNeuron(1, 128, kernel_size=3, padding=1,
                                                        bias=True, v_threshold= args.V, v_reset= None),
                                        nn.BatchNorm2d(128),
                                        )

        self.conv = nn.Sequential(
                                #nn.MaxPool2d(kernel_size=2, stride=2),
                                nn.AvgPool2d(kernel_size=2, stride=2),
                                Conv2d_IFNeuron(128, 128, kernel_size=3,
                                                padding=1, bias=True, v_threshold= args.V, v_reset= None),
                                nn.BatchNorm2d(128),
                                nn.AvgPool2d(kernel_size=2, stride=2),
                                #nn.MaxPool2d(kernel_size=2, stride=2),
                                )
        self.fc = nn.Sequential(
                                nn.Flatten(),
                                #layer.Dropout(0.5),
                                IFNeuron(128 * 7 * 7, 128 * 4 * 4, 
                                        bias=True, v_threshold= args.V, v_reset= None),
                                #layer.Dropout(0.5),
                                nn.Linear(128 * 4 * 4, 10, bias=True),
                                )

    def forward(self, x):
        x = self.static_conv(x)
        
        out_spikes_counter = self.fc(self.conv(x))
        for t in range(1, self.T):
            out_spikes_counter += self.fc(self.conv(x))

        return out_spikes_counter 

net_lenet5 = nn.Sequential(
                        Conv2d_IFNeuron(1, 6, kernel_size=5, padding=2, v_threshold= args.V, v_reset= None),
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        Conv2d_IFNeuron(6, 16, kernel_size=5, v_threshold= args.V, v_reset= None),
                        nn.AvgPool2d(kernel_size=2, stride=2),                                                                                                               
                        nn.Flatten(),
                        IFNeuron(16 * 5 * 5, 120, v_threshold= args.V, v_reset= None), 
                        IFNeuron(120, 84, v_threshold= args.V, v_reset= None),
                        nn.Linear(84, 10)
                        )

net_aaai = nn.Sequential(
                        Conv2d_IFNeuron(1, 20, kernel_size=5, bias=True, v_threshold= args.V, v_reset= None),
                        nn.BatchNorm2d(20),
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        Conv2d_IFNeuron(20, 50, kernel_size=5, bias=True, v_threshold= args.V, v_reset= None),
                        nn.BatchNorm2d(50),
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        nn.Flatten(),
                        #layer.Dropout(0.5),
                        IFNeuron(800, 100, bias=True, v_threshold= args.V, v_reset= None),
                        #layer.Dropout(0.5),
                        nn.Linear(100, 10, bias=True),
                        )
def dataset_loader(name, Trans = transforms.ToTensor()):
    print('加载数据集中')
    if name == 'mnist':
        train_set = datasets.MNIST(root = args.data_dir,
                                train = True,
                                transform = Trans,
                                download = True,
                                )
        test_set = datasets.MNIST(root = args.data_dir,
                                train = False,
                                transform = Trans,
                                download = True,
                                )
        train_iter = torch.utils.data.DataLoader(train_set, pin_memory = True,
                    batch_size = args.batch_size, shuffle = True, num_workers = args.num_worker)
        test_iter = torch.utils.data.DataLoader(test_set, pin_memory = True,
                    batch_size = args.batch_size, shuffle = False, num_workers = args.num_worker)
    if name == 'Fashionmnist':
        train_set = datasets.FashionMNIST(root = args.data_dir,
                                train = True,
                                transform = Trans,
                                download = True,
                                )
        test_set = datasets.FashionMNIST(root = args.data_dir,
                                train = False,
                                transform = Trans,
                                download = True,
                                )
        train_iter = torch.utils.data.DataLoader(train_set, pin_memory = True,
                    batch_size = args.batch_size, shuffle = True, num_workers = args.num_worker)
        test_iter = torch.utils.data.DataLoader(test_set, pin_memory = True,
                    batch_size = args.batch_size, shuffle = False, num_workers = args.num_worker)
    return train_iter, test_iter


train_data, test_data = dataset_loader(name = args.dataset)

def trainer(net, encoder, T, train_data_loader, test_data_loader, 
            device, epochs):
    print("-" * 20)
    print('训练已经开始:')
    begin = time.time()
    net = net.to(device)
    #net = nn.DataParallel(net, device_ids=[3, 4])
    train_times = 0
    max_test_accuracy = 0
    test_accs = []
    train_accs = []
    optimizer = None
    if args.opt == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr = args.lr, momentum = args.momentum)
    elif args.opt == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    lr_scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min= args.eta_min) # 余弦退火学习率调整
    #writer = SummaryWriter(log_dir= args.out_dir, purge_step= 0)
    for epoch in range(epochs):
        print("Epoch {}:".format(epoch))
        print("Training...")
        train_correct_sum = 0
        train_sum = 0
        net.train()
        loss_num = 0
        loss_test = 0
        for img, label in tqdm(train_data_loader):
            img = img.to(device)
            label = label.to(device)
            label_one_hot = F.one_hot(label, 10).float()
            optimizer.zero_grad()
            # 运行T个时长，out_spikes_counter是shape=[batch_size, 10]的tensor
            # 记录整个仿真时长内，输出层的10个神经元的脉冲发放次数
            if type(encoder) == encoding.PoissonEncoder:
                for t in range(T):
                    if t == 0:
                        out_spikes_membranes = net(encoder(img).float32())
                    else:
                        out_spikes_membranes += net(encoder(img).float32())
            elif encoder == Sigma_Delta_Encoder:
                #ssss = Dual_mormalization(img, mapping = 'linear')
                encoders = encoder(Dual_mormalization(img, mapping = 'linear') , t = T, dim = 0)
                #encoders = encoder(img , t = T, dim = 0)
                for t in range(T):
                    if t == 0:
                            out_spikes_membranes = net(encoders[t])
                    else:
                            out_spikes_membranes += net(encoders[t])
            elif encoder == 'layer':
                out_spikes_membranes = net(img)
            # out_spikes_counter / T 得到输出层10个神经元在仿真时长内的脉冲发放频率
            # 在本网络(SSDSNN)中，最后一层不发放脉冲
            out_spikes_counter_frequency = out_spikes_membranes / T
            #out_spikes_counter_frequency = F.softmax(F.relu(out_spikes_membranes - args.V), dim = 1)
            # 损失函数为输出层神经元的脉冲发放频率，与真实类别的MSE
            # 这样的损失函数会使，当类别i输入时，输出层中第i个神经元的脉冲发放频率趋近1，而其他神经元的脉冲发放频率趋近0
            loss = F.mse_loss(out_spikes_counter_frequency, label_one_hot)
            loss.requires_grad_(True)   # 解决element 0 of tensors does not require grad and does not have a grad_fn的鬼bug
            loss_num += loss
            loss.backward()
            optimizer.step()
            # 优化一次参数后，需要重置网络的状态，因为SNN的神经元是有“记忆”的
            functional.reset_net(net)

            # 正确率的计算方法如下。认为输出层中脉冲发放频率最大的神经元的下标i是分类结果
            train_correct_sum += (out_spikes_counter_frequency.max(1)[1] == label.to(device)).float().sum().item()
            train_sum += label.numel()
            train_batch_accuracy = (out_spikes_counter_frequency.max(1)[1] == label.to(device)).float().mean().item()
            #writer.add_scalar('train_accuracy', train_accuracy, train_times)
            train_accs.append(train_batch_accuracy)
            train_times += 1
        print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
        lr_scheduler1.step()
        train_accuracy = train_correct_sum / train_sum
        #writer.add_scalar('train_loss', loss, epoch)
        #writer.add_scalar('train_accuracy', train_accuracy, epoch)
        print("Testing...")
        net.eval()
        with torch.no_grad():
                # 每遍历一次全部数据集，就在测试集上测试一次
                test_correct_sum = 0
                test_sum = 0
                for img, label in tqdm(test_data_loader):
                    img = img.to(device)
                    label = label.to(device)
                    label_one_hot = F.one_hot(label, 10).float()
                    if type(encoder) == encoding.PoissonEncoder:
                        for t in range(T):
                            if t == 0:
                                out_spikes_membranes = net(encoder(img).float32())
                            else:
                                out_spikes_membranes += net(encoder(img).float32())
                    elif encoder == Sigma_Delta_Encoder:
                        encoders = encoder(Dual_mormalization(img, mapping = 'linear') , t = T, dim = 0)
                        #encoders = encoder(img , t = T, dim = 0)
                        for t in range(T):
                            if t == 0:
                                    out_spikes_membranes = net(encoders[t])
                            else:
                                    out_spikes_membranes += net(encoders[t])
                    elif encoder == 'layer':
                        out_spikes_membranes = net(img)  # 输出为(batch_size, class_number)
                    out_spikes_counter_frequency = out_spikes_membranes / T
                    #out_spikes_counter_frequency = F.softmax(F.relu(out_spikes_membranes - args.V), dim = 1)
                    loss = F.mse_loss(out_spikes_counter_frequency, label_one_hot)
                    loss_test += loss
                    test_correct_sum += (out_spikes_membranes.max(1)[1] == label.to(device)).float().sum().item()
                    test_sum += label.numel()
                    functional.reset_net(net)
                test_accuracy = test_correct_sum / test_sum
                test_accs.append(test_accuracy)
                max_test_accuracy = max(max_test_accuracy, test_accuracy)
                #writer.add_scalar('test_accuracy', test_accuracy, epoch)
        print(f'train_loss: {loss_num:.4f}, test_loss：{loss_test : .4f}')
        print("Epoch {}: train_acc = {}%, test_acc={}%, max_test_acc={}%, train_times={}".format(epoch, train_accuracy * 100, test_accuracy * 100, max_test_accuracy * 100, train_times))
        print()
    end = time.time()
    print(f'本次训练花费时间为：{(end - begin):.3f}秒')

print('准备训练')

trainer(net = net_aaai,  encoder = Sigma_Delta_Encoder, T = args.T,
        train_data_loader = train_data, test_data_loader = test_data,
        device = args.device, epochs = args.epochs)

   
