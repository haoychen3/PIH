import datetime
import math
import shutil

import dateutil.tz
import os
import os.path as osp
from shutil import copyfile, copytree
import glob
import time
import random
import torch

from ConSinGAN.config import get_arguments
import ConSinGAN.functions as functions

import pathlib


def get_scale_factor(opt):
    opt.scale_factor = 1.0
    num_scales = math.ceil((math.log(math.pow(opt.min_size / (min(real.shape[2], real.shape[3])), 1), opt.scale_factor_init))) + 1
    opt.scale_factor_init = opt.scale_factor
    if opt.num_training_scales > 0:
        while num_scales > opt.num_training_scales:
            opt.scale_factor_init = opt.scale_factor_init - 0.01
            num_scales = math.ceil((math.log(math.pow(opt.min_size / (min(real.shape[2], real.shape[3])), 1), opt.scale_factor_init))) + 1
    return opt.scale_factor_init


# noinspection PyInterpreter
if __name__ == '__main__':
    '''
    first secret image
    '''
    parser = get_arguments()
    parser.add_argument('--input_name', help='input image name for training', default="")
    parser.add_argument('--naive_img', help='naive input image  (harmonization or editing)', default="")
    parser.add_argument('--gpu', type=int, help='which GPU to use', default=0)
    parser.add_argument('--train_mode', default='generation',
                        choices=['generation', 'retarget', 'harmonization', 'editing', 'animation'],
                        help="generation, retarget, harmonization, editing, animation")
    parser.add_argument('--lr_scale', type=float, help='scaling of learning rate for lower stages', default=0.1)
    parser.add_argument('--train_stages', type=int, help='how many stages to use for training', default=6)

    parser.add_argument('--fine_tune', action='store_true', help='whether to fine tune on a given image', default=0)
    parser.add_argument('--model_dir', help='model to be used for fine tuning (harmonization or editing)', default="")

    opt = parser.parse_args()
    opt = functions.post_config(opt)

    '''
    second secret image
    '''
    parser_2 = get_arguments()
    parser_2.add_argument('--input_name', help='input image name for training', default="")
    parser_2.add_argument('--naive_img', help='naive input image  (harmonization or editing)', default="")
    parser_2.add_argument('--gpu', type=int, help='which GPU to use', default=0)
    parser_2.add_argument('--train_mode', default='generation',
                              choices=['generation', 'retarget', 'harmonization', 'editing', 'animation'],
                              help="generation, retarget, harmonization, editing, animation")
    parser_2.add_argument('--lr_scale', type=float, help='scaling of learning rate for lower stages', default=0.1)
    parser_2.add_argument('--train_stages', type=int, help='how many stages to use for training', default=6)

    parser_2.add_argument('--fine_tune', action='store_true', help='whether to fine tune on a given image',
                              default=0)
    parser_2.add_argument('--model_dir', help='model to be used for fine tuning (harmonization or editing)',
                              default="")

    opt_2 = parser_2.parse_args()
    opt_2 = functions.post_config(opt_2)

    '''
    cover
    '''
    new_parser = get_arguments()
    new_parser.add_argument('--input_name', help='input image name for training', default="")
    new_parser.add_argument('--naive_img', help='naive input image  (harmonization or editing)', default="")
    new_parser.add_argument('--gpu', type=int, help='which GPU to use', default=0)
    new_parser.add_argument('--train_mode', default='generation',
                        choices=['generation', 'retarget', 'harmonization', 'editing', 'animation'],
                        help="generation, retarget, harmonization, editing, animation")
    new_parser.add_argument('--lr_scale', type=float, help='scaling of learning rate for lower stages', default=0.1)
    new_parser.add_argument('--train_stages', type=int, help='how many stages to use for training', default=6)

    new_parser.add_argument('--fine_tune', action='store_true', help='whether to fine tune on a given image', default=0)
    new_parser.add_argument('--model_dir', help='model to be used for fine tuning (harmonization or editing)', default="")

    new_opt = new_parser.parse_args()
    new_opt = functions.post_config(new_opt)

    path1 = pathlib.Path('./Image-coco/secret')
    files1 = list(sorted(path1.glob('*')))

    path2 = pathlib.Path('./Image-div2k/secret')
    files2 = list(sorted(path2.glob('*')))

    path3 = pathlib.Path('./Image-coco/cover')
    files3 = list(sorted(path3.glob('*')))

    for i in range(len(files1)):
        opt.input_name = str(files1[i])
        opt_2.input_name = str(files2[i])
        new_opt.input_name = str(files3[i])

        if opt.fine_tune:
            _gpu = opt.gpu
            _model_dir = opt.model_dir
            _timestamp = opt.timestamp
            _naive_img = opt.naive_img
            _niter = opt.niter

            opt = functions.load_config(opt)

            opt.gpu = _gpu
            opt.model_dir = _model_dir
            opt.start_scale = opt.train_stages - 1
            opt.timestamp = _timestamp
            opt.fine_tune = True
            opt.naive_img = _naive_img
            opt.niter = _niter

        if not os.path.exists(opt.input_name):
            print("Image does not exist: {}".format(opt.input_name))
            print("Please specify a valid image.")
            exit()

        if torch.cuda.is_available():
            torch.cuda.set_device(opt.gpu)

        if opt.train_mode == "generation" or opt.train_mode == "retarget" or opt.train_mode == "animation":
            if opt.train_mode == "animation":
                opt.min_size = 20
            from ConSinGAN.training_generation import *
        elif opt.train_mode == "harmonization" or opt.train_mode == "editing":
            if opt.fine_tune:
                if opt.model_dir == "":
                    print("Model for fine tuning not specified.")
                    print("Please use --model_dir to define model location.")
                    exit()
                else:
                    if not os.path.exists(opt.model_dir):
                        print("Model does not exist: {}".format(opt.model_dir))
                        print("Please specify a valid model.")
                        exit()
                if not os.path.exists(opt.naive_img):
                    print("Image for harmonization/editing not found: {}".format(opt.naive_img))
                    exit()
            from ConSinGAN.training_harmonization_editing import *

        dir2save = functions.generate_dir2save(opt)

        if osp.exists(dir2save):
            print('Trained model already exist: {}'.format(dir2save))
            exit()

        # create log dir
        try:
            os.makedirs(dir2save)
        except OSError:
            pass

        # save hyperparameters and code files
        with open(osp.join(dir2save, 'parameters.txt'), 'w') as f:
            for o in opt.__dict__:
                f.write("{}\t-\t{}\n".format(o, opt.__dict__[o]))
        current_path = os.path.dirname(os.path.abspath(__file__))
        for py_file in glob.glob(osp.join(current_path, "*.py")):
            try:
                copyfile(py_file, osp.join(dir2save, py_file.split("/")[-1]))
            except shutil.SameFileError:
                pass
        copytree(osp.join(current_path, "ConSinGAN"), osp.join(dir2save, "ConSinGAN"))

        # train model
        print("Training model ({})".format(dir2save))
        start = time.time()
        train(opt, opt_2, new_opt)
        end = time.time()
        elapsed_time = end - start
        print("Time for training: {} seconds".format(elapsed_time))
