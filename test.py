import argparse
import torch
import os
import numpy as np
import utils
import skimage.color as sc
import cv2
from model import  esrt

# Testing settings

parser = argparse.ArgumentParser(description='ESRT')
parser.add_argument("--test_hr_folder", type=str, default='Test_Datasets/Set5/',
                    help='the folder of the target images')
parser.add_argument("--test_lr_folder", type=str, default='Test_Datasets/Set5_LR/x2/',
                    help='the folder of the input images')
parser.add_argument("--output_folder", type=str, default='results/Set5/x2')
parser.add_argument("--checkpoint", type=str, default='checkpoints/IMDN_x2.pth',
                    help='checkpoint folder to use')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use cuda')
parser.add_argument("--upscale_factor", type=int, default=2,
                    help='upscaling factor')
parser.add_argument("--is_y", action='store_true', default=True,
                    help='evaluate on y channel, if False evaluate on RGB channels')
opt = parser.parse_args()

print(opt)
def forward_chop(model, x, shave=10, min_size=60000):
    scale = 4#self.scale[self.idx_scale]
    n_GPUs = 1#min(self.n_GPUs, 4)
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    lr_list = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        sr_list = []
        for i in range(0, 4, n_GPUs):
            lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
            sr_batch = model(lr_batch)
            sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
    else:
        sr_list = [
            forward_chop(model, patch, shave=shave, min_size=min_size) \
            for patch in lr_list
        ]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = x.new(b, c, h, w)
    output[:, :, 0:h_half, 0:w_half] \
        = sr_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output
cuda = opt.cuda
device = torch.device('cuda' if cuda else 'cpu')

filepath = opt.test_hr_folder
if filepath.split('/')[-2] == 'Set5' or filepath.split('/')[-2] == 'Set14':
    ext = '.bmp'
else:
    ext = '.png'

filelist = utils.get_list(filepath, ext=ext)
psnr_list = np.zeros(len(filelist))
ssim_list = np.zeros(len(filelist))
time_list = np.zeros(len(filelist))

model =  esrt.ESRT(upscale = opt.upscale_factor)#
model_dict = utils.load_state_dict(opt.checkpoint)
model.load_state_dict(model_dict, strict=False)#True)

i = 0
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

for imname in filelist:
    im_gt = cv2.imread(imname, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB
    im_gt = utils.modcrop(im_gt, opt.upscale_factor)
    im_l = cv2.imread(opt.test_lr_folder + imname.split('/')[-1].split('.')[0] + 'x' + str(opt.upscale_factor) + ext, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB
    if len(im_gt.shape) < 3:
        im_gt = im_gt[..., np.newaxis]
        im_gt = np.concatenate([im_gt] * 3, 2)
        im_l = im_l[..., np.newaxis]
        im_l = np.concatenate([im_l] * 3, 2)
    im_input = im_l / 255.0
    im_input = np.transpose(im_input, (2, 0, 1))
    im_input = im_input[np.newaxis, ...]
    im_input = torch.from_numpy(im_input).float()

    if cuda:
        model = model.to(device)
        im_input = im_input.to(device)

    with torch.no_grad():
        start.record()
        out = forward_chop(model, im_input) #model(im_input)
        end.record()
        torch.cuda.synchronize()
        time_list[i] = start.elapsed_time(end)  # milliseconds

    out_img = utils.tensor2np(out.detach()[0])
    crop_size = opt.upscale_factor
    cropped_sr_img = utils.shave(out_img, crop_size)
    cropped_gt_img = utils.shave(im_gt, crop_size)
    if opt.is_y is True:
        im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
        im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
    else:
        im_label = cropped_gt_img
        im_pre = cropped_sr_img
    psnr_list[i] = utils.compute_psnr(im_pre, im_label)
    ssim_list[i] = utils.compute_ssim(im_pre, im_label)


    output_folder = os.path.join(opt.output_folder,
                                 imname.split('/')[-1].split('.')[0] + 'x' + str(opt.upscale_factor) + '.png')

    if not os.path.exists(opt.output_folder):
        os.makedirs(opt.output_folder)

    cv2.imwrite(output_folder, out_img[:, :, [2, 1, 0]])
    i += 1


print("Mean PSNR: {}, SSIM: {}, TIME: {} ms".format(np.mean(psnr_list), np.mean(ssim_list), np.mean(time_list)))
