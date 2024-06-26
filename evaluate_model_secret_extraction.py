import os
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

from ConSinGAN.config import get_arguments
import ConSinGAN.functions as functions
import ConSinGAN.models as models
from ConSinGAN.imresize import imresize, imresize_to_shape


def make_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


def generate_samples(netG, reals_shapes, noise_amp, reconstruct=False, n=25):
    if reconstruct:
        reconstruction = netG(fixed_noise, reals_shapes, noise_amp)
        functions.save_image('{}/extracted_secret.png'.format(dir2save), reconstruction.detach())
        return reconstruction

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--model_dir', help='path to saved model', default='./test-model')
    parser.add_argument('--gpu', type=int, help='which GPU', default=0)
    parser.add_argument('--num_samples', type=int, help='which GPU', default=50)

    opt = parser.parse_args()
    _gpu = opt.gpu
    __model_dir = opt.model_dir
    opt = functions.load_config(opt)
    opt.gpu = _gpu
    opt.model_dir = __model_dir

    if torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu)
        opt.device = "cuda:{}".format(opt.gpu)

    dir2save = os.path.join(opt.model_dir, "Evaluation")
    make_dir(dir2save)

    print("Loading models...")
    netG = torch.load('%s/G.pth' % opt.model_dir, map_location="cuda:{}".format(torch.cuda.current_device()))
    fixed_noise = torch.load('%s/fixed_noise.pth' % opt.model_dir, map_location="cuda:{}".format(torch.cuda.current_device()))
    reals = torch.load('%s/reals.pth' % opt.model_dir, map_location="cuda:{}".format(torch.cuda.current_device()))
    noise_amp = torch.load('%s/noise_amp.pth' % opt.model_dir, map_location="cuda:{}".format(torch.cuda.current_device()))
    reals_shapes = [r.shape for r in reals]

    print("Extracting Secret...")
    with torch.no_grad():

        generate_samples(netG, reals_shapes, noise_amp, reconstruct=True)


    print("Done. Results saved at: {}".format(dir2save))

