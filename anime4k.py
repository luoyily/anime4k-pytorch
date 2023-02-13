import os

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.utils.data as Data
from torchvision.io import read_image
from torchvision.io import ImageReadMode

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Configs :
device = torch.device("cuda")
# per steps
save_interval = 2000
log_interval = 100
eval_interval = 1000
# model save dir
save_path = './model/'
# tensorboard log dir
log_dir = './model/log/'
global_steps = 0
# dataset path: directories containing jpg or png images
train_data_path = 'E:/dataset/sr/train'
eval_data_path = 'E:/dataset/sr/eval'
batch_size = 48

class SRData(Data.Dataset):
    def __init__(self, folder, scale=2, gt_size=(256, 256), eval_mode=False):
        self.scale = scale
        self.folder = folder
        self.gt_size = gt_size
        self.eval_mode = eval_mode
        self.imgs = [os.path.join(self.folder, f) for f in os.listdir(
            folder) if (f.endswith('.png') or f.endswith('.jpg'))]

    def add_data(self, folder):
        to_add = [os.path.join(folder, f) for f in os.listdir(
            folder) if (f.endswith('.png') or f.endswith('.jpg'))]
        self.imgs += to_add

    def __getitem__(self, index):

        img_fn = self.imgs[index]
        img = read_image(img_fn, mode=ImageReadMode.RGB)
        img = img.to('cuda')/255
        img_size = (img.shape[1], img.shape[2])
        if self.eval_mode:
            # if img_size != self.gt_size:
            #     img = transforms.Resize((int(img_size[0]/self.scale),int(img_size[1]/self.scale)))(img)
            if (img_size[0] % self.scale != 0 or img_size[1] % self.scale != 0):
                img = transforms.CenterCrop(
                    ((img_size[0]//self.scale)*self.scale, (img_size[1]//self.scale)*self.scale))(img)
            gt_img = img
            lq_img = transforms.Resize((int(img_size[0]/self.scale),int(img_size[1]/self.scale)))(img)
        else:
            # preprocess
            # random crop
            if img_size != self.gt_size:
                img = transforms.RandomCrop((self.gt_size))(img)
        # data argument
            # img = transforms.ColorJitter(0.3,0.3,0.3,0.3)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.RandomVerticalFlip()(img)
            gt_img = img
            # low quality img
            lq_img = transforms.Resize(
                (int(self.gt_size[0]/self.scale), int(self.gt_size[1]/self.scale)))(img)
        return lq_img, gt_img

    def __len__(self):
        return len(self.imgs)


sr_datasets = SRData(train_data_path)
dataloaders = torch.utils.data.DataLoader(sr_datasets, batch_size=batch_size, shuffle=True, num_workers=0)
eval_datasets = SRData(eval_data_path, eval_mode=True)


class CReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = nn.ReLU()

    def forward(self, x):
        out = torch.cat((self.m(x), self.m(-x)), dim=1)
        return out


class Anime4K(nn.Module):
    def __init__(self, input_channel=3, conv_channel=4, block_depth=4, upscale=2):
        super().__init__()
        self.upscale = upscale

        self.conv_block = nn.ModuleList()
        # first
        self.conv_0 = nn.Conv2d(
            input_channel, conv_channel, (3, 3), stride=1, padding=1)
        self.crelu_0 = CReLU()
        # block
        for i in range(block_depth-1):
            self.conv_block.append(
                nn.Conv2d(conv_channel*2, conv_channel, (3, 3), stride=1, padding=1))
            self.conv_block.append(CReLU())
        # other
        self.conv_last = nn.Conv2d(
            block_depth * conv_channel*2, input_channel * upscale * upscale, (1, 1), stride=1, padding=0)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear')
        out = self.conv_0(x)
        out = self.crelu_0(out)
        depth_list = []
        depth_list.append(out)
        for i in range(len(self.conv_block)):
            out = self.conv_block[i](out)
            if i % 2 != 0:
                depth_list.append(out)
        # cat
        out = torch.cat(depth_list, dim=1)
        # last conv
        out = self.conv_last(out)
        # DepthToSpace
        out = self.upsampler(out)
        # add
        out = torch.add(base, out)
        return out


# ITU-R BT.709 conversion
def rgb_tensor2ycbcr_tensor(imgs):
    r = imgs[:, 0, :, :]
    g = imgs[:, 1, :, :]
    b = imgs[:, 2, :, :]
    y = r * 0.2126+g*0.7152+b*0.0722
    cb = r*(-0.1146)+g*(-0.3854)+b*0.5
    cr = r*0.5+g*(-0.4542)+b*(-0.0458)
    return (y, cb, cr)


# ITU-R BT.601 conversion
def rgb_tensor2ycbcr_tensor_bt601(imgs):
    r = imgs[:, 0, :, :]
    g = imgs[:, 1, :, :]
    b = imgs[:, 2, :, :]
    y = (16 + r*65.481 + g*128.553 + b*24.966)/255
    cb = (128 + r*(-37.797) - g*(74.203) + b*112)/255
    cr = (128 + r*112 - g*(93.786) - b*18.214)/255
    return (y, cb, cr)


class YCbCrLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, y_pred, y_true):
        pred_y, pred_cb, pred_cr = rgb_tensor2ycbcr_tensor(y_pred)
        true_y, true_cb, true_cr = rgb_tensor2ycbcr_tensor(y_true)
        # pred_y, pred_cb, pred_cr = rgb_tensor2ycbcr_tensor_bt601(y_pred)
        # true_y, true_cb, true_cr = rgb_tensor2ycbcr_tensor_bt601(y_true)
        y_error = self.mse(pred_y, true_y) * 0.5
        cb_error = self.mse(pred_cb, true_cb) * 0.25
        cr_error = self.mse(pred_cr, true_cr) * 0.25
        total_error = y_error+cb_error+cr_error
        total_error = total_error * 255
        total_error.requires_grad_(True)
        return total_error


def calculate_psnr(img, img2):
    mse = torch.mean((img - img2)**2)
    return 10. * torch.log10(1. / (mse + 1e-8))


def train_model(model, criterion, optimizer, start_epoch=0, num_epochs=20):
    global global_steps
    model.to(device)
    writer = SummaryWriter(log_dir=log_dir)
    temp_loss = 0
    temp_psnr = 0
    for epoch in range(start_epoch, num_epochs):
        # Iterate over data.
        with tqdm(total=len(dataloaders)) as pbar:
            for lq_img, gt_img in dataloaders:
                if global_steps % save_interval == 0 and global_steps != 0:
                    # print(f'Save model: {global_steps} Steps.')
                    torch.save(model.state_dict(), os.path.join(
                        save_path, f'{global_steps}.pth'))

                lq_img = lq_img.to(device)
                gt_img = gt_img.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = model(lq_img)
                    loss = criterion(outputs, gt_img)
                    temp_loss = loss
                    loss.backward()
                    optimizer.step()
                    pbar.set_description(
                        f'[{epoch}/{num_epochs - 1}E][{global_steps}][Loss:{round(float(loss),6)}][Psnr:{round(float(temp_psnr),2)}]')
                    pbar.update(1)

                if global_steps % log_interval == 0:
                    # write log
                    writer.add_scalar(f'Loss', temp_loss, global_steps)
                if global_steps % eval_interval == 0 and global_steps != 0:
                    model.eval()
                    # print(f'Start eval:{global_steps}')
                    lq_img, gt_img = eval_datasets[0]
                    lq_img = lq_img.to(device)
                    gt_img = gt_img.to(device)
                    pred_img = model(torch.unsqueeze(lq_img, dim=0))
                    img_save = transforms.ToPILImage()(pred_img[0].clamp_(0, 1))
                    img_save.save(os.path.join(log_dir, f'{global_steps}.png'))
                    psnr = calculate_psnr(pred_img[0].clamp_(0, 1), gt_img)
                    temp_psnr = psnr
                    writer.add_scalar(f'Psnr', psnr, global_steps)
                    model.train()
                global_steps += 1
    return model


if __name__ == '__main__':
    model = Anime4K(block_depth=8)
    # model.load_state_dict(torch.load('./model/400000.pth'))
    criterion = YCbCrLoss()
    optimizer = torch.optim.RAdam(model.parameters(), lr=0.0001)
    train_model(model, criterion, optimizer, num_epochs=10000)
