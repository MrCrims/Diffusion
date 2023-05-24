from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import diffusion_utils
from unet.unet_model import Unet
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision import transforms as T
from torch.optim import Adam
from pathlib import Path
from accelerate import Accelerator
from ema_pytorch import EMA
import math
import os
from PIL import Image

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

class SimpleDiffusion(nn.Module):
    def __init__(self,
                 backbone_model,
                 image_size,
                 device,
                 timesteps=1000,
                 sampling_timesteps=None,
                 beta_schedule='linear',
                 ) -> None:
        super().__init__()
        self.model = backbone_model
        self.channels = self.model.channels
        self.image_size = image_size
        self.num_timesteps = timesteps
        beta_schedule_fn = diffusion_utils.cosine_beta_schedule

        betas = beta_schedule_fn(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)
        # print(alphas_cumprod_prev.shape)
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32)) # 将参数保留在模型中
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)#t时刻的参数
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)#t-1时刻的参数

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # 将采样的方差设置为一个固定的与beta有关的常数使得可训练参数就在均值里面
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        snr = alphas_cumprod / (1 - alphas_cumprod)
        self.loss_weight = snr/(snr+1)
        self.loss_fn = nn.L2Loss().to(device)
    
    #本质上都是那个重参数化的公式转换的 ： xt = \mu x0 + \beta e
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            diffusion_utils.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            diffusion_utils.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (diffusion_utils.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            diffusion_utils.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def q_posterior(self, x_start, x_t, t):
        # 计算后验的扩散概率q(xt-1|xt,x0)
        posterior_mean = (
            diffusion_utils.extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            diffusion_utils.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = diffusion_utils.extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = diffusion_utils.extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def model_prediction(self,x,t):
        # 目标是预测x0
        model_out = self.model(x)
        x_start = model_out
        pred_noise = self.predict_noise_from_start(x,t,x_start)
        
        return pred_noise,x_start
    
    def p_mean_variance(self,x,t):
        pred_noise,x_start = self.model_prediction(x,t)
        x_start.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start
    
    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond = None):
        # 用的是重参数化的方法生成的样本
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        # xt-1 = \mu(xt,t) + sqrt(\sigma) z
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, return_all_timesteps = False):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for t in reversed(range(0, self.num_timesteps)):
            # self_cond = x_start if self.self_condition else None
            self_cond = None
            img, x_start = self.p_sample(img, t, self_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = (ret+1)*0.5
        return ret
    
    @torch.no_grad()
    def sample(self, batch_size = 16, return_all_timesteps = False):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop
        return sample_fn((batch_size, channels, image_size, image_size), return_all_timesteps = return_all_timesteps)
    
    def q_sample(self,x_start,t,noise=None):
        noise = default(noise,lambda: torch.randn_like(x_start))


        return (
            diffusion_utils.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            diffusion_utils.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self,x_start,t,noise=None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        #from x0 to xT
        x = self.q_sample(x_start = x_start, t = t, noise = noise)#从x0到xT
        
        model_out = self.model(x)
        loss = self.loss_fn(model_out,x_start)
        loss  = loss * diffusion_utils.extract(self.loss_weight,t,loss.shape)
        return loss.mean()
    
    def forward(self,img):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        t = torch.randint(0,self.num_timesteps,(b,),device=device).long()

        img = 2 * img - 1
        return self.p_losses(img,t)

class Dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

class Trainer(object):
    def __init__(self,
        diffusion_model,
        folder,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-3,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        fp16 = False,
        split_batches = True,
        convert_image_to = None,
        calculate_fid = True,
        inception_block_idx = 2048
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )
        self.accelerator.native_amp = amp

        self.model = diffusion_model
        self.channels = diffusion_model.channels
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.ds = Dataset(folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)
        print(self.ds[5].shape,len(self.ds))
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = 1)

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)
        
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
    
    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            # 'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
    
    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        for i in range(self.step,self.train_num_steps):

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                # pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        all_images = torch.cat(all_images_list, dim = 0)

                        save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))
                        self.save(milestone)

                        # whether to calculate fid

                        # if exists(self.inception_v3):
                        #     fid_score = self.fid_score(real_samples = data, fake_samples = all_images)
                        #     accelerator.print(f'fid_score: {fid_score}')

                # pbar.update(1)

        accelerator.print('training complete')
        return
