import torch, os, math, logging
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.nn.init import normal, constant
from model.resnet import resnet18
# from resnet import resnet18
from PIL import Image
from transformers import AutoModel
import torch.nn.functional as F
from model.transformer import Transformer, PosEmb

logger = logging.getLogger(__name__)

class BaselineLSTM(nn.Module):
    def __init__(self, args):
        super(BaselineLSTM, self).__init__()
        self.args = args
        self.img_feature_dim = 256  # the dimension of the CNN feature to represent each frame
        self.base_model = resnet18(pretrained=False)
        self.base_model.fc2 = nn.Linear(1000, self.img_feature_dim)
        self.lstm = nn.LSTM(self.img_feature_dim, self.img_feature_dim, bidirectional=True, num_layers=2, batch_first=True)
        self.last_layer1 = nn.Linear(2*self.img_feature_dim, 128)
        self.last_layer2 = nn.Linear(128, 2)

        for param in self.parameters():
            param.requires_grad = True
        
        self._init_parameters()
        
        self.load_checkpoint()

    def forward(self, input):
        N, D, C, H, W = input.shape
        base_out = self.base_model(input.view(N*D, C, H, W))
        base_out = base_out.view(N, D, self.img_feature_dim)
        lstm_out, _ = self.lstm(base_out)
        lstm_out = lstm_out[:,3,:]
        output = self.last_layer1(lstm_out)
        output = self.last_layer2(output)
        return output

    def load_checkpoint(self):
        if self.args.checkpoint is not None:
            if os.path.exists(self.args.checkpoint):
                logger.info(f'loading checkpoint {self.args.checkpoint}')
                state = torch.load(self.args.checkpoint, map_location=f'cuda:{self.args.rank}')
                if 'module' in list(state["state_dict"].keys())[0]:
                    state_dict = { k[7:]: v for k, v in state["state_dict"].items() }
                else:
                    state_dict = state["state_dict"]
                self.load_state_dict(state_dict)
        
    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class GazeLSTM(nn.Module):
    def __init__(self, args):
        super(GazeLSTM, self).__init__()
        self.args = args
        self.img_feature_dim = 256  # the dimension of the CNN feature to represent each frame
        self.base_model = resnet18(pretrained=True)
        self.base_model.fc2 = nn.Linear(1000, self.img_feature_dim)
        self.lstm = nn.LSTM(self.img_feature_dim, self.img_feature_dim,bidirectional=True,num_layers=2,batch_first=True)
        # The linear layer that maps the LSTM with the 2 outputs
        self.last_layer1 = nn.Linear(2 * self.img_feature_dim, 128)
        self.last_layer2 = nn.Linear(128, 2)
        self.load_checkpoint()

    def forward(self, input):
        base_out = self.base_model(input.view((-1, 3) + input.size()[-2:]))
        base_out = base_out.view(input.size(0),7,self.img_feature_dim)
        lstm_out, _ = self.lstm(base_out)
        lstm_out = lstm_out[:,3,:]
        output = self.last_layer1(lstm_out)
        output = self.last_layer2(output).view(-1,2)
        return output

    def load_checkpoint(self):
        if self.args.checkpoint is not None:
            if os.path.exists(self.args.checkpoint):
                logger.info(f'loading checkpoint {self.args.checkpoint}')
                map_loc = f'cuda:{self.args.rank}' if torch.cuda.is_available() else 'cpu'
                state = torch.load(self.args.checkpoint, map_location=map_loc)
                if 'module' in list(state["state_dict"].keys())[0]:
                    state_dict = { k[7:]: v for k, v in state["state_dict"].items() }
                else:
                    state_dict = state["state_dict"]
                if 'gaze360' in self.args.checkpoint:
                    state_dict.pop('last_layer.weight')
                    state_dict.pop('last_layer.bias')
                self.load_state_dict(state_dict, strict=self.args.eval)
            else:
                logger.info(f'checkpoint {self.args.checkpoint} not exists')

class ViTLSTM(nn.Module):
    def __init__(self, args):
        super(ViTLSTM, self).__init__()
        self.args = args
        path = "/data/zhanggl/sda/xj/Ego4d/InternViT-6B-448px-V1-5"

        self.img_feature_dim = 1000
        self.fc1 = nn.Linear(3200, 1000)
        self.lstm = nn.LSTM(self.img_feature_dim, self.img_feature_dim, bidirectional=True,num_layers=2,batch_first=True)
        self.last_layer1 = nn.Linear(2 * self.img_feature_dim, 1024)
        self.last_layer2 = nn.Linear(1024, 512)
        self.last_layer3 = nn.Linear(512, 512)
        self.last_layer4 = nn.Linear(512, 2)
        self.drop = nn.Dropout(0.5)
        self.load_checkpoint()

    def forward(self, input):
        base_out = self.fc1(input)
        lstm_out, _ = self.lstm(base_out)
        lstm_out = lstm_out[:,3,:]
        output = self.drop(F.relu(self.last_layer1(lstm_out)))
        output = self.drop(F.relu(self.last_layer2(output)))
        output = self.drop(F.relu(self.last_layer3(output)))
        output = self.last_layer4(output).view(-1,2)
        return output
    
    def load_checkpoint(self):
        if self.args.checkpoint is not None:
            if os.path.exists(self.args.checkpoint):
                logger.info(f'loading checkpoint {self.args.checkpoint}')
                map_loc = f'cuda:{self.args.rank}' if torch.cuda.is_available() else 'cpu'
                state = torch.load(self.args.checkpoint, map_location=map_loc)
                if 'module' in list(state["state_dict"].keys())[0]:
                    state_dict = { k[7:]: v for k, v in state["state_dict"].items() }
                else:
                    state_dict = state["state_dict"]
                if 'gaze360' in self.args.checkpoint:
                    state_dict.pop('last_layer.weight')
                    state_dict.pop('last_layer.bias')
                self.load_state_dict(state_dict, strict=self.args.eval)
            else:
                logger.info(f'checkpoint {self.args.checkpoint} not exists')