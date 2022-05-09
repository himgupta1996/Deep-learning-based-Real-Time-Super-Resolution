from __future__ import print_function
import argparse
from math import log10
import os
import torch
from torch._C import dtype
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import get_test_set 
import time
import torch.backends.cudnn as cudnn
import cv2
import math
import sys
import datetime
from utils import Logger
import numpy as np
import torchvision.utils as vutils
from arch import RRN
import time

parser = argparse.ArgumentParser(description='PyTorch RRN Example')
parser.add_argument('--scale', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--testbatchsize', type=int, default=1, help='testing batch size')
parser.add_argument('--threads', type=int, default=32, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=0, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--cuda',default=True, type=bool)
parser.add_argument('--layer', type=int, default=10, help='network layer')
parser.add_argument('--test_dir',type=str,default='/home/panj/data/Vid4')
#parser.add_argument('--test_dir',type=str,default='/home/panj/data/udm10')
#parser.add_argument('--test_dir',type=str,default='/home/panj/data/SPMC_test')
parser.add_argument('--save_test_log', type=str,default='./log/test')
parser.add_argument('--pretrain', type=str, default='./model/RRN-5L.pth')
parser.add_argument('--image_out', type=str, default='./out/')
opt = parser.parse_args()
gpus_list = range(opt.gpus)
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
print(opt)


def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (KB):', size/1e3)
    os.remove('temp.p')
    return size

class QuantizedRNN(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedRNN, self).__init__()

        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant_x = torch.quantization.QuantStub()
        self.quant_h = torch.quantization.QuantStub()
        self.quant_o = torch.quantization.QuantStub()

        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant_h = torch.quantization.DeQuantStub()
        self.dequant_o = torch.quantization.DeQuantStub()

        # FP32 model
        self.model_fp32 = model_fp32

    def forward(self, x, h, o):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x_q = self.quant_x(x)
        h_q = self.quant_h(h)
        o_q = self.quant_o(o)

        new_h_q, new_o_q = self.model_fp32(x_q, h_q, o_q)

        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        new_h = self.dequant_h(new_h_q)
        new_o = self.dequant_o(new_o_q)

        return new_h, new_o


def main():
    n_c = 128
    n_b = 5

    old_rrn = RRN(opt.scale, n_c, n_b) # initial filter generate network

    print('===> load pretrained model')
    if os.path.isfile(opt.pretrain):
        state_dict = torch.load(opt.pretrain, map_location=lambda storage, loc: storage)

        for key in list(state_dict.keys()):
            state_dict[key.replace('module.', '')] = state_dict.pop(key)

        old_rrn.load_state_dict(state_dict)
        print('===> pretrained model is load')
    else:
        raise Exception('pretrain model is not exists')

    print('===> Loading test Datasets')
    PSNR_avg = 0
    SSIM_avg = 0
    count = 0
    scene_list = ['foliage', 'calendar', 'city', 'walk'] # Vid4

    for scene_name in scene_list:
        test_set = get_test_set(opt.test_dir, opt.scale, scene_name)
        test_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testbatchsize, shuffle=False, drop_last=False)
        print('===> DataLoading Finished')

        print('===> Creating Quantized Model for: {}'.format(scene_name))
        quant_rrn = QuantizedRNN(old_rrn)
        quant_rrn.eval()
        
        # attach a global qconfig, which contains information about what kind
        # of observers to attach. Use 'fbgemm' for server inference and
        # 'qnnpack' for mobile inference. Other quantization configurations such
        # as selecting symmetric or assymetric quantization and MinMax or L2Norm
        # calibration techniques can be specified here.
        quant_rrn.qconfig = torch.quantization.get_default_qconfig('fbgemm')

        # Fuse the activations to preceding layers, where applicable.
        # This needs to be done manually depending on the model architecture.
        # Common fusions include `conv + relu` and `conv + batchnorm + relu`
        # rrn_fused = torch.quantization.fuse_modules(rrn, [['conv', 'relu']])

        # Prepare the model for static quantization. This inserts observers in
        # the model that will observe activation tensors during calibration.
        # rrn_prepared = torch.quantization.prepare(rrn_fused)
        rrn_prepared = torch.quantization.prepare(quant_rrn)

        # calibrate the prepared model to determine quantization parameters for activations
        # in a real world setting, the calibration would be done with a representative dataset
        sample_batch = next(iter(test_loader))
        sample_input = sample_batch[0][:, :, :2, :, :]
        sample_o = sample_batch[1][:, :, 0, :, :]
        
        # sample_input = torch.rand((1, 3, 2, 124, 184), dtype=torch.float32)
        # sample_o = torch.rand((1, 3, 124 * 4, 184 * 4), dtype=torch.float32)
        height = sample_input.size(3)
        width = sample_input.size(4)
        sample_h = torch.rand((1, n_c, height, width), dtype=torch.float32)

        rrn_prepared(sample_input, sample_h, sample_o)

        # Convert the observed model to a quantized model. This does several things:
        # quantizes the weights, computes and stores the scale and bias value to be
        # used with each activation tensor, and replaces key operators with quantized
        # implementations.
        rrn_int8 = torch.quantization.convert(rrn_prepared)
        rrn_int8.eval()

        # compare the sizes
        f = print_size_of_model(quant_rrn, "fp32")
        q = print_size_of_model(rrn_int8, "int8")
        # print("{0:.2f} times smaller".format(f/q))

        PSNR, SSIM = test(test_loader, rrn_int8, opt.scale, scene_name, n_c)
        PSNR_avg += PSNR
        SSIM_avg += SSIM
        count += 1

    PSNR_avg = PSNR_avg/len(scene_list)
    SSIM_avg = SSIM_avg/len(scene_list)
    print('==> Average PSNR = {:.6f}'.format(PSNR_avg))
    print('==> Average SSIM = {:.6f}'.format(SSIM_avg))

def test(test_loader, rrn, scale, scene_name, n_c):
    train_mode = False

    count = 0
    PSNR = 0
    SSIM = 0
    PSNR_t = 0
    SSIM_t = 0
    out = []
    for image_num, data in enumerate(test_loader):
        x_input, target = data[0], data[1]
        B, _, T, _ ,_ = x_input.shape
        T = T - 1 # not include the padding frame
        with torch.no_grad():
            x_input = Variable(x_input)
            target = Variable(target)
            # t0 = time.time()
            total_time = 0
            init = True
            for i in range(T):
                if init:
                    init_temp = torch.zeros_like(x_input[:,0:1,0,:,:])
                    init_o = init_temp.repeat(1, 3, scale, scale)
                    init_h = init_temp.repeat(1, n_c, 1, 1)

                    t0 = time.time()
                    h, prediction = rrn(x_input[:,:,i:i+2,:,:], init_h, init_o)
                    t1 = time.time()

                    out.append(prediction)
                    init = False
                else:
                    t0 = time.time()
                    h, prediction = rrn(x_input[:,:,i:i+2,:,:], h, prediction)
                    t1 = time.time()
                    
                    total_time += t1 - t0

                    out.append(prediction)
                    
        print("===> Timer: %.4f sec." % (total_time))
        
        prediction = torch.stack(out, dim=2)
        count += 1
        prediction = prediction.squeeze(0).permute(1,2,3,0) # [T,H,W,C]
        prediction = prediction.numpy()[:,:,:,::-1] # tensor -> numpy, rgb -> bgr 
        target = target.squeeze(0).permute(1,2,3,0) # [T,H,W,C]
        target = target.numpy()[:,:,:,::-1] # tensor -> numpy, rgb -> bgr
        target = crop_border_RGB(target, 8)
        prediction = crop_border_RGB(prediction, 8)
        for i in range(T):
            save_img(prediction[i], scene_name, i)
            # test_Y______________________
            prediction_Y = bgr2ycbcr(prediction[i])
            target_Y = bgr2ycbcr(target[i])
            prediction_Y = prediction_Y * 255
            target_Y = target_Y * 255
            # test_RGB _______________________________
            #prediction_Y = prediction[i] * 255
            #target_Y = target[i] * 255
            # ________________________________
            # calculate PSNR and SSIM
            print('PSNR: {:.6f} dB, \tSSIM: {:.6f}'.format(calculate_psnr(prediction_Y, target_Y), calculate_ssim(prediction_Y, target_Y)))
            PSNR += calculate_psnr(prediction_Y, target_Y)
            SSIM += calculate_ssim(prediction_Y, target_Y)
            out.append(calculate_psnr(prediction_Y, target_Y))
        print('===>{} PSNR = {}'.format(scene_name, PSNR / T))
        print('===>{} SSIM = {}'.format(scene_name, SSIM / T))
        PSNR_t += PSNR / T
        SSIM_t += SSIM / T

    return PSNR_t, SSIM_t

def save_img(prediction, scene_name, image_num):
    save_dir = os.path.join(opt.image_out, systime)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    image_dir = os.path.join(save_dir, '{}_{:03}'.format(scene_name, image_num+1) + '.png')
    cv2.imwrite(image_dir, prediction*255, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    Output:
        type is same as input
        unit8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def crop_border_Y(prediction, shave_border=0):
    prediction = prediction[shave_border:-shave_border, shave_border:-shave_border]
    return prediction

def crop_border_RGB(target, shave_border=0):
    target = target[:,shave_border:-shave_border, shave_border:-shave_border,:]
    return target

def calculate_psnr(prediction, target):
    # prediction and target have range [0, 255]
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')



if __name__=='__main__':
    main()
