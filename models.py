from torchvision.models.mobilenetv2 import MobileNetV2 as MyModel # 2.35M
import timm, logging
import torch
from utils import freeze_model, count_parameters
from arcfaceloss import *
# model = timm.create_model('vit_tiny_r_s16_p8_224', pretrained=True)
# print(timm.list_models('*vit*'))

NUM_CLASSES = 102

class RexNet100(nn.Module):
    def __init__(self, enet_type, out_dim, load_pretrained=True):
        super(RexNet100, self).__init__()
        self.enet = timm.create_model('rexnet_100', pretrained=False, num_classes=NUM_CLASSES)
        # for param in self.enet.parameters():
        #     param.requires_grad = False
        
        self.feat = nn.Linear(self.enet.head.fc.in_features, 512)
        self.swish = Swish_module()
        self.metric_classify = ArcMarginProduct_subcenter(512, out_dim)
        self.enet.head.fc = nn.Identity()

    def forward(self, x):
        return self.metric_classify(self.swish(self.feat(self.enet(x))))

def getModel():
    # model = timm.create_model('vit_tiny_r_s16_p8_224', pretrained=True, num_classes=NUM_CLASSES)
    # logging.info(count_parameters(model))
    # freeze_model(model, fine_tuning=False)
    # logging.info(count_parameters(model))

    model = timm.create_model('rexnet_100', pretrained=False, num_classes=NUM_CLASSES) # 3.64M
    logging.info(f'Model params: {count_parameters(model)/1e6:.3f}M')
    return model