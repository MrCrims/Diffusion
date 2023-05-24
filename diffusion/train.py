from model import *
import torchvision
datapath = "F:\WM Group\working\data\\test"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

backbone_model = Unet(3,3)
diffusion = SimpleDiffusion(backbone_model,32,device)

trainer = Trainer(
    diffusion,
    datapath,
    train_batch_size=32
)

trainer.train()