import os
import os.path as osp
import math
import argparse
import yaml
import time
import traceback
import torch
import sys
import json
import sys
#sys.path.append('.')
from joblib import Parallel, delayed
sys.path.append('/work/hgupta_umass_edu/EVISR_new/Deep-learning-based-Real-Time-Super-Resolution/VI')
import cv2
import itertools
import numpy as np
import megengine as mge
import megengine.functional as F
from model.RIFE import Model
from contextlib import suppress
import skimage as skim
import dill as pickle
from joblib.externals.loky import set_loky_pickler
set_loky_pickler("dill")
import concurrent.futures
model_vi = Model(simply_infer=True)
model_vi.load_model('/work/hgupta_umass_edu/EVISR_new/Deep-learning-based-Real-Time-Super-Resolution/VI/train_log')
model_vi.eval()

from data import create_dataloader, prepare_data
from models import define_model
from models.networks import define_generator
from metrics.metric_calculator import MetricCalculator
from metrics.model_summary import register, profile_model
from utils import base_utils, data_utils


def train(opt):
    # logging
    logger = base_utils.get_logger('base')
    logger.info('{} Options {}'.format('='*20, '='*20))
    base_utils.print_options(opt, logger)

    # create data loader
    train_loader = create_dataloader(opt, dataset_idx='train')

    # create downsampling kernels for BD degradation
    kernel = data_utils.create_kernel(opt)

    # create model 
    model = define_model(opt)

    # training configs
    total_sample = len(train_loader.dataset)
    iter_per_epoch = len(train_loader)
    total_iter = opt['train']['total_iter']
    total_epoch = int(math.ceil(total_iter / iter_per_epoch))
    start_iter, iter = opt['train']['start_iter'], 0

    test_freq = opt['test']['test_freq']
    log_freq = opt['logger']['log_freq']
    ckpt_freq = opt['logger']['ckpt_freq']

    logger.info('Number of training samples: {}'.format(total_sample))
    logger.info('Total epochs needed: {} for {} iterations'.format(
        total_epoch, total_iter))

    # train
    for epoch in range(total_epoch):
        for data in train_loader:
            # update iter
            iter += 1
            curr_iter = start_iter + iter
            if iter > total_iter:
                logger.info('Finish training')
                break

            # update learning rate
            model.update_learning_rate()

            # prepare data
            data = prepare_data(opt, data, kernel)

            # train for a mini-batch
            model.train(data)

            # update running log
            model.update_running_log()

            # log
            if log_freq > 0 and iter % log_freq == 0:
                # basic info
                msg = '[epoch: {} | iter: {}'.format(epoch, curr_iter)
                for lr_type, lr in model.get_current_learning_rate().items():
                    msg += ' | {}: {:.2e}'.format(lr_type, lr)
                msg += '] '

                # loss info
                log_dict = model.get_running_log()
                msg += ', '.join([
                    '{}: {:.3e}'.format(k, v) for k, v in log_dict.items()])

                logger.info(msg)

            # save model
            if ckpt_freq > 0 and iter % ckpt_freq == 0:
                model.save(curr_iter)

            # evaluate performance
            if test_freq > 0 and iter % test_freq == 0:
                # setup model index
                model_idx = 'G_iter{}'.format(curr_iter)

                # for each testset
                for dataset_idx in sorted(opt['dataset'].keys()):
                    # use dataset with prefix `test`
                    if not dataset_idx.startswith('test'):
                        continue

                    ds_name = opt['dataset'][dataset_idx]['name']
                    logger.info(
                        'Testing on {}: {}'.format(dataset_idx, ds_name))

                    # create data loader
                    test_loader = create_dataloader(opt, dataset_idx=dataset_idx)

                    # define metric calculator
                    metric_calculator = MetricCalculator(opt)

                    # infer and compute metrics for each sequence
                    for data in test_loader:
                        # fetch data
                        lr_data = data['lr'][0]
                        seq_idx = data['seq_idx'][0]
                        frm_idx = [frm_idx[0] for frm_idx in data['frm_idx']]

                        # infer
                        hr_seq = model.infer(lr_data)  # thwc|rgb|uint8

                        # save results (optional)
                        if opt['test']['save_res']:
                            res_dir = osp.join(
                                opt['test']['res_dir'], ds_name, model_idx)
                            res_seq_dir = osp.join(res_dir, seq_idx)
                            data_utils.save_sequence(
                                res_seq_dir, hr_seq, frm_idx, to_bgr=True)

                        # compute metrics for the current sequence
                        true_seq_dir = osp.join(
                            opt['dataset'][dataset_idx]['gt_seq_dir'], seq_idx)
                        metric_calculator.compute_sequence_metrics(
                            seq_idx, true_seq_dir, '', pred_seq=hr_seq)

                    # save/print metrics
                    if opt['test'].get('save_json'):
                        # save results to json file
                        json_path = osp.join(
                            opt['test']['json_dir'], '{}_avg.json'.format(ds_name))
                        metric_calculator.save_results(model_idx, json_path, override=True)
                    else:
                        # print directly
                        metric_calculator.display_results()


def test(opt):
    # logging
    logger = base_utils.get_logger('base')
    if opt['verbose']:
        logger.info('{} Configurations {}'.format('=' * 20, '=' * 20))
        base_utils.print_options(opt, logger)

    # infer and evaluate performance for each model
    for load_path in opt['model']['generator']['load_path_lst']:
        # setup model index
        model_idx = osp.splitext(osp.split(load_path)[-1])[0]
        
        # log
        logger.info('=' * 40)
        logger.info('Testing model: {}'.format(model_idx))
        logger.info('=' * 40)

        # create model
        opt['model']['generator']['load_path'] = load_path
        try:
            model = define_model(opt)
        except Exception as e:
            raise Exception(e)

        result = {}
        # for each test dataset
        for dataset_idx in sorted(opt['dataset'].keys()):
            # use dataset with prefix `test`
            print("Computing for dataset %s" % (dataset_idx))
            if not dataset_idx.startswith('test'):
                continue

            ds_name = opt['dataset'][dataset_idx]['name']
            logger.info('Testing on {}: {}'.format(dataset_idx, ds_name))

            result[ds_name] = {}

            # define metric calculator
            try:
                metric_calculator = MetricCalculator(opt)
            except:
                print('No metirc need to compute!')

            # create data loader
            test_loader = create_dataloader(opt, dataset_idx=dataset_idx)

            # infer and store results for each sequence
            for i, data in enumerate(test_loader):

                total_time_taken = 0

                total_sequence_frames = len(data['frm_idx'])

                no_segments = math.ceil(total_sequence_frames/opt['chunk_size'])
                seq_idx = data['seq_idx'][0]
                result[ds_name][seq_idx] = {}

                fps = []
                vsr_time = []
                vi_time = []

                input_resolution = data['lr'].shape[1]*data['lr'].shape[2]

                print(f"Sequence: {seq_idx}")

                for k in range(no_segments):
                    print(f"===Processing {k+1}/{no_segments} chunk===")
                    lr_data = data['lr'][0][k*opt['chunk_size']:k*opt['chunk_size']+opt['chunk_size']]
                    no_of_frames = len(lr_data)
                    
                    frm_idx = [frm_idx[0] for frm_idx in data['frm_idx'][k*opt['chunk_size']:k*opt['chunk_size']+opt['chunk_size']]]

                    to_interpolate_ids = list(range(1, len(lr_data)-1, 2))
                    to_interpolate_ids_set = set(to_interpolate_ids)
                    to_superresolute_ids = [i for i in range(0, len(lr_data)) if i not in to_interpolate_ids_set]

                    lr_alt_seq = lr_data[to_superresolute_ids]
                    superresoluted_frame_ids = [frm_idx[i] for i in to_superresolute_ids]
                    interpolated_frame_ids = [frm_idx[i] for i in to_interpolate_ids]
                    
                    print(f"The size of the lr sequence is {lr_alt_seq.shape[2]}")

                    #print(f"superresoluted frames: {superresoluted_frame_ids}, interpolated frames: {interpolated_frame_ids}")
                    #print(f"The shape of tensor which have to be super resoluted is {lr_alt_seq.shape}")

                    ##super resoluting alternative frames
                    start_time = time.time()

                    vsr_start_time = time.time()
                    hr_alt_seq = model.infer(lr_alt_seq)
                    vsr_end_time = time.time()
                    #print(f"The shape of tensor which has been super resoluted is {hr_alt_seq.shape}")

                    ##interpolating middle frames
                    #hr_alt_seq1 = hr_alt_seq[..., ::-1]
                    #print(f"shape of super resoluted frames are {hr_alt_seq.shape}")
                    vi_start_time = time.time()
                    interpolated_frames = interpolate(hr_alt_seq[..., ::-1], to_superresolute_ids, to_interpolate_ids)
                    vi_end_time = time.time()
                    #print(f"The length of tensor which has been interpolated is {len(interpolated_frames)}")

                    end_time = time.time()

                    time_taken = end_time - start_time
                    total_time_taken += time_taken

                    fps.append(no_of_frames/time_taken)
                    vsr_time.append((no_of_frames, vsr_start_time, vsr_end_time, vsr_end_time-vsr_start_time))
                    vi_time.append((no_of_frames, vi_start_time, vi_end_time, vi_end_time-vi_start_time))

                    print(f"===End processing for chunck {k+1}/{no_segments}===")

                    # save results (optional)
                    if opt['test']['save_res']:
                        res_dir = osp.join(opt['test']['res_dir'], ds_name, model_idx)
                        res_seq_dir = osp.join(res_dir, seq_idx)
                        data_utils.save_sequence(res_seq_dir, hr_alt_seq, superresoluted_frame_ids, to_bgr=True)
                        #print(f"The interpolated frames are {interpolated_frames}")
                        data_utils.save_sequence(res_seq_dir, interpolated_frames, interpolated_frame_ids, to_bgr=False)

                    # compute metrics for the current sequence
                true_seq_dir = osp.join(opt['dataset'][dataset_idx]['gt_seq_dir'], seq_idx)

                try:
                    res_dir = osp.join(opt['test']['res_dir'], ds_name, model_idx)
                    res_seq_dir = osp.join(res_dir, seq_idx)
                    metric_calculator.compute_sequence_metrics(seq_idx, true_seq_dir, res_seq_dir, pred_seq=None)
                except Exception as e:
                    print("ERROR in calculating sequence metrics %s" % (str(e)))
                    print('No metirc need to compute!')

                average_fps = total_sequence_frames/total_time_taken

                print(f"The total time taken to process {total_sequence_frames} frames is: {total_time_taken}, and average fps is: {average_fps}")
                print(f"FPS for different chunks are {fps}")
                result[ds_name][seq_idx]['total_time_taken'] = total_time_taken
                result[ds_name][seq_idx]['total_frames'] = total_sequence_frames
                result[ds_name][seq_idx]['individual_fps'] = fps
                result[ds_name][seq_idx]['average_fps'] = average_fps
                result[ds_name][seq_idx]['vsr_time'] = vsr_time
                result[ds_name][seq_idx]['vi_time'] = vi_time
                result[ds_name][seq_idx]['input_resolution'] = input_resolution

            # save/print metrics and result
            try:
                if opt['test'].get('save_json'):
                    # save results to json file
                    json_path = osp.join(
                        opt['test']['json_dir'], '{}_avg.json'.format(ds_name))
                    metric_calculator.save_results(model_idx, json_path, override=True)
                    
                else:
                    # print directly
                    result = metric_calculator.display_results(result, ds_name)

            except Exception as e:
                print(f"The exception is %s" % (str(e)))
                print('No metirc need to save!')

            logger.info('-' * 40)
        
        print(f"The result is {result}")

        with open(osp.join(opt['test']['json_dir'], '{}_{}_{}.json'.format(model_idx, ds_name, opt['chunk_size'])), 'w') as outfile:
            json.dump(result, outfile)

    # logging
    logger.info('Finish testing')
    logger.info('=' * 40)

def interpolate_middle_frame(frame1, frame2, index, index_frame_mapping):
    # global model_vi
    # global temp_global
    # print(f"The temp global is {temp_global}")
    # import megengine.functional as F
    # import megengine as mge
    #"The mapping is {index_frame_mapping}")
    #print(f"The index is {index}")
   # print(f"Entering the fucntion with index {index}")
    I0 = F.expand_dims(mge.Tensor(frame1.transpose(2, 0, 1)) / 255. , 0)
    I1 = F.expand_dims(mge.Tensor(frame2.transpose(2, 0, 1)) / 255. , 0)
    # print(f"The index is {index}")
    n, c, h, w = I0.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = ((0, 0), (0, 0), (0, ph - h), (0, pw - w))
    I0 = F.nn.pad(I0, padding)
    I1 = F.nn.pad(I1, padding)
    pred = model_vi.inference(I0, I1)
            
    #pred = pred[:, :, pad: -pad]
    out = (pred[0] * 255).numpy().transpose(1, 2, 0)[:h, :w]
    #index_frame_mapping[index] = frame1
    #print(f"I am index {index}")

    return out

def interpolate(hr_seq, hr_seq_org_indexes, to_interpolate_ids):
    frame_list = []
    index_frame_mapping = {}
    # executor = concurrent.futures.ProcessPoolExecutor(10)
    # futures = [executor.submit(interpolate_middle_frame, frames[0], frames[1], indexes[0]+1, index_frame_mapping) for frames, indexes in zip(list(base_utils.pairwise(hr_seq)), list(base_utils.pairwise(hr_seq_org_indexes))) if indexes[0]+2 == indexes[1]]
    # print("Before wait")
    # concurrent.futures.wait(futures)
    # print("In interpolate")
    # print(index_frame_mapping.keys())
    
    #out = Parallel(n_jobs=-1, require='sharedmem')(delayed(interpolate_middle_frame)(frames[0], frames[1], indexes[0]+1, index_frame_mapping) for frames, indexes in zip(list(base_utils.pairwise(hr_seq)), list(base_utils.pairwise(hr_seq_org_indexes))) if indexes[0]+2 == indexes[1])
    i=0
    while i<len(hr_seq)-1:
        if hr_seq_org_indexes[i+1] == hr_seq_org_indexes[i]+2:
            out = interpolate_middle_frame(hr_seq[i], hr_seq[i+1], hr_seq_org_indexes[i]+1, index_frame_mapping)
            frame_list.append(out)
        i+=1
    # for id in to_interpolate_ids:
    #     frame_list.append(index_frame_mapping[id])
    return frame_list

def profile(opt, lr_size, test_speed=False):
    # logging
    logger = base_utils.get_logger('base')
    logger.info('{} Model Information {}'.format('='*20, '='*20))
    base_utils.print_options(opt['model']['generator'], logger)

    # basic configs
    scale = opt['scale']
    device = torch.device(opt['device'])

    # create model
    net_G = define_generator(opt).to(device)

    # get dummy input
    dummy_input_dict = net_G.generate_dummy_input(lr_size)
    for key in dummy_input_dict.keys():
        dummy_input_dict[key] = dummy_input_dict[key].to(device)

    # profile
    register(net_G, dummy_input_dict)
    gflops, params = profile_model(net_G)

    logger.info('-' * 40)
    logger.info('Super-resolute data from {}x{}x{} to {}x{}x{}'.format(
        *lr_size, lr_size[0], lr_size[1]*scale, lr_size[2]*scale))
    logger.info('Parameters (x10^6): {:.3f}'.format(params/1e6))
    logger.info('FLOPs (x10^9): {:.3f}'.format(gflops))
    logger.info('-' * 40)

    # test running speed
    if test_speed:
        n_test = 30
        tot_time = 0

        for i in range(n_test):
            start_time = time.time()
            with torch.no_grad():
                _ = net_G(**dummy_input_dict)
            end_time = time.time()
            tot_time += end_time - start_time

        logger.info('Speed (FPS): {:.3f} (averaged for {} runs)'.format(
            n_test / tot_time, n_test))
        logger.info('-' * 40)


if __name__ == '__main__':
    logger = base_utils.get_logger('base')

    # ----------------- parse arguments ----------------- #
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='directory of the current experiment')
    parser.add_argument('--mode', type=str, required=True,
                        help='which mode to use (train|test|profile)')
    parser.add_argument('--model', type=str, required=True,
                        help='which model to use (FRVSR|TecoGAN)')
    parser.add_argument('--opt', type=str, required=True,
                        help='path to the option yaml file')
    parser.add_argument('--gpu_id', type=int, default=-1,
                        help='GPU index, -1 for CPU')
    parser.add_argument('--lr_size', type=str, default='3x256x256',
                        help='size of the input frame')
    parser.add_argument('--test_speed', action='store_true',
                        help='whether to test the actual running speed')
    parser.add_argument('--chunk-size', type=int, default=30,
                        help='number of frames in one chunck of a video')
    ##Todo: Take input how many frames to interpolate in between
    args = parser.parse_args()


    # ----------------- get options ----------------- #
    print(args.exp_dir)
    with open(osp.join(args.exp_dir, args.opt), 'r') as f:
        opt = yaml.load(f.read(), Loader=yaml.FullLoader)


    # ----------------- general configs ----------------- #
    # experiment dir
    opt['exp_dir'] = args.exp_dir
    opt['chunk_size'] = args.chunk_size
    opt['save_json'] = True

    # random seed
    base_utils.setup_random_seed(opt['manual_seed'])

    # logger
    base_utils.setup_logger('base')
    opt['verbose'] = opt.get('verbose', False)

    # device
    if args.gpu_id >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            opt['device'] = 'cuda'
        else:
            opt['device'] = 'cpu'
    else:
        opt['device'] = 'cpu'

    # ----------------- train ----------------- #
    if args.mode == 'train':
        # setup paths
        base_utils.setup_paths(opt, mode='train')

        # run
        opt['is_train'] = True
        train(opt)

    # ----------------- test ----------------- #
    elif args.mode == 'test':
        # setup paths
        base_utils.setup_paths(opt, mode='test')

        # run
        opt['is_train'] = False
        with suppress(OSError):
            print('i am in testing')
            test(opt)

    # ----------------- profile ----------------- #
    elif args.mode == 'profile':
        lr_size = tuple(map(int, args.lr_size.split('x')))

        # run
        profile(opt, lr_size, args.test_speed)

    else:
        raise ValueError(
            'Unrecognized mode: {} (train|test|profile)'.format(args.mode))
