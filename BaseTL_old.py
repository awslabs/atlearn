from dtl.model_zoo.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from dtl.model_zoo.alexnet import alexnet
from dtl.model_zoo.vgg import vgg11, vgg13, vgg16, vgg19
from dtl.model_zoo.densenet import densenet121, densenet161, densenet169, densenet201
from dtl.model_zoo.squeezenet import squeezenet1_0, squeezenet1_1
from dtl.model_zoo.vit import vit_b_16, vit_b_32, vit_l_16, vit_l_32
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime


class BaseTL(nn.Module):
    def __init__(self, path, network='resnet18', freeze=True, gpu_id=-1, save_name="default", save_path=""):
        super(BaseTL, self).__init__()
        if gpu_id >= 0 and torch.cuda.is_available():
            self.device = torch.device("cuda:{}".format(gpu_id))
        elif gpu_id >= 0 and not torch.cuda.is_available():
            self.device = torch.device("cpu")
            print("No GPU is found! It will use CPU directly.")
        else:
            self.device = torch.device("cpu")
        self.path = path
        self.save_name = save_name
        self.save_path = save_path

        instance_gen = globals()[network]
        self.base_network = instance_gen(pretrained=True).to(device=self.device)
        self.fts_dim = self.base_network.feat_dim
        for param in self.base_network.parameters():
            param.requires_grad = freeze
        self.fcn = None
        self.optimizer = None

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight.data)
        #         nn.init.constant_(m.bias.data, 0)

    def forward(self, inputs):
        inputs = self.cnn(inputs)
        inputs = self.fcn(inputs)
        return inputs

    def get_optimizer(self, lr):
        params_to_update = []
        for name, param in self.base_network.named_parameters():
            if param.requires_grad:
                params_to_update.append({'params': param, 'lr': 0.1 * lr})
        for name, param in self.fcn.named_parameters():
            if param.requires_grad:
                params_to_update.append({'params': param, 'lr': lr})
        self.optimizer = optim.Adam(params_to_update)

    def update(self):
        raise NotImplementedError

    def mixup(self, inputs, labels, alpha=1.0):
        '''
        A simple data augmentation approach from
        * Zhang, Hongyi, et al. "mixup: Beyond Empirical Risk Minimization." In ICLR. 2018.
        :param inputs: input images (input-level mixup)
        :param alpha: a hyper-parameter for generating the mixture magnitude
        :return: mixed images, and generated mixup factor
        '''
        lam = np.random.beta(alpha, alpha)
        index = torch.randperm(inputs.shape[0])
        inputs_mix = lam * inputs + (1 - lam) * inputs[index]
        return inputs_mix, lam, labels, labels[index]

    def traced_model(self):
        model = torch.nn.Sequential()
        model.append(self.base_network)
        model.append(self.fcn)
        save_traced_network = torch.jit.trace(model.eval(), torch.rand(1, 3, 224, 224))  # .eval() is required here
        save_traced_network.save('{}/{}.pt'.format(self.save_path, self.save_name))

    def save_checkpoint(self, epoch, training_loss):
        date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
        model = torch.nn.Sequential()
        model.append(self.base_network)
        model.append(self.fcn)
        model.eval()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': training_loss,
        }, "../checkpoint/{}_{}_epoch_{}_{}.pt".format(self.save_name, self.base_network_name, epoch, date))

    @staticmethod
    def print_text(text):
        print("\033[91m {}\033[00m".format(text))

    @staticmethod
    def set_random_seed(seed=0):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
