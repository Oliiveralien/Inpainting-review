import os
import random
import time
import shutil
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from scripts.trainer import Trainer
from data.dataset_loader import Dataset
from utils.tools import get_config
from utils.logger import get_logger
from model.mask import mask_image

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml',
                    help="training configuration")
parser.add_argument('--seed', type=int, default="2019", help='manual seed')
parser.add_argument('--iteration', type=int, default="0", help='resume iteration number')


def main():
    args = parser.parse_args()
    config = get_config(args.config)

    # CUDA configuration
    cuda = config['cuda']
    device_ids = config['gpu_ids']
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
        device_ids = list(range(len(device_ids)))
        config['gpu_ids'] = device_ids
        cudnn.benchmark = True

    # Configure checkpoint path
    checkpoint_path = os.path.join('checkpoints',
                                   config['dataset_name'],
                                   config['mask_type'] + '_' + config['expname'])
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    shutil.copy(args.config, os.path.join(checkpoint_path, os.path.basename(args.config)))
    writer = SummaryWriter(log_dir=checkpoint_path)
    logger = get_logger(checkpoint_path)  # get logger and configure it at the first call

    logger.info(f"Arguments: {args}")
    # Set random seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    logger.info(f"Random seed: {args.seed}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed_all(args.seed)

    # Log the configuration
    logger.info(f"Configuration: {config}")

    try:  # for unexpected error logging
        # Load the dataset
        logger.info(f"Training on dataset: {config['dataset_name']}")
        train_dataset = Dataset(data_path=config['train_data_path'],
                                with_subfolder=config['data_with_subfolder'],
                                image_shape=config['image_shape'],
                                random_crop=config['random_crop'])

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=config['batch_size'],
                                                   shuffle=True,
                                                   num_workers=config['num_workers'])


        # Define the trainer
        trainer = Trainer(config)
        logger.info(f"\n{trainer.netG}")
        logger.info(f"\n{trainer.globalD}")

        if cuda:
            trainer = nn.parallel.DataParallel(trainer, device_ids=device_ids)
            trainer_module = trainer.module
        else:
            trainer_module = trainer

        # Get the resume iteration to restart training
        start_iteration = trainer_module.resume(config['checkpoint_dir'], args.iteration) if config['resume'] else 1

        iterable_train_loader = iter(train_loader)

        time_count = time.time()

        for iteration in range(start_iteration, config['niter'] + 1):
            try:
                ground_truth = iterable_train_loader.next()
            except StopIteration:
                iterable_train_loader = iter(train_loader)
                ground_truth = iterable_train_loader.next()

            # Prepare the inputs
            x, mask = mask_image(ground_truth, config)
            if cuda:
                x = x.cuda()
                mask = mask.cuda()
                ground_truth = ground_truth.cuda()

            ###### Forward pass ######
            losses, coarse_result, inpainted_result = trainer(x, mask, ground_truth)
            # Scalars from different devices are gathered into vectors
            for k in losses.keys():
                if not losses[k].dim() == 0:
                    losses[k] = torch.mean(losses[k])

            ###### Backward pass ######
            # Update D
            trainer_module.optimizer_d.zero_grad()
            losses['d'] = losses['d_loss_rel'] +  losses['d_loss_loren']
            losses['d'].backward(retain_graph=True)
            trainer_module.optimizer_d.step()

            # Update G
            trainer_module.optimizer_g.zero_grad()
            losses['g'] = losses['l1'] * config['l1_loss_alpha']  +  losses['g_loss_rel']  + losses['g_loss_loren']
            losses['g'].backward()
            trainer_module.optimizer_g.step()

            # Log and visualization
            log_losses = ['l1', 'g', 'd' ]
            if iteration % config['print_iter'] == 0:
                time_count = time.time() - time_count
                speed = config['print_iter'] / time_count
                speed_msg = 'speed: %.2f batches/s ' % speed
                time_count = time.time()

                message = 'Iter: [%d/%d] ' % (iteration, config['niter'])
                for k in log_losses:
                    v = losses[k]
                    writer.add_scalar(k, v, iteration)
                    message += '%s: %.6f ' % (k, v)
                message += speed_msg
                logger.info(message)
            def denorm(x):
                out = (x + 1) / 2  # [-1,1] -> [0,1]
                return out.clamp_(0, 1)

            if iteration % (config['viz_iter']) == 0:
                ims = torch.cat([x, coarse_result, inpainted_result, ground_truth], dim=3)
                writer.add_image('raw_masked_coarse_refine', denorm(ims), iteration)
            # if iteration % (config['save_image']) ==0:
            #     viz_max_out = config['viz_max_out']
            #     if x.size(0) > viz_max_out:
            #         viz_images = torch.stack([x[:viz_max_out], coarse_result[:viz_max_out], inpainted_result[:viz_max_out],
            #                                   ], dim=1)
            #     else:
            #         viz_images = torch.stack([x, coarse_result, inpainted_result], dim=1)
            #     viz_images = viz_images.view(-1, *list(x.size())[1:])
            #     vutils.save_image(viz_images,
            #                       '%s/niter_%03d.png' % (checkpoint_path, iteration),
            #                       nrow=3 * 4,
            #                       normalize=True)


            # Save the model
            if iteration % config['snapshot_save_iter'] == 0:
                trainer_module.save_model(checkpoint_path)

    except Exception as e:  # for unexpected error logging
        logger.error(f"{e}")
        raise e


if __name__ == '__main__':
    main()
