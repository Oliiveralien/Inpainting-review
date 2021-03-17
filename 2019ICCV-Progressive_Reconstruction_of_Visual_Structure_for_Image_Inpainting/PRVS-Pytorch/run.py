import argparse
import os
from model import PRVSModel
from dataset import Dataset
from torch.utils.data import DataLoader

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../../Dataset/Pairs_street_view/paris_auged')
    parser.add_argument('--mask_root', type=str, default='../../Dataset/irregular_mask/testing_mask_dataset_auged')
    parser.add_argument('--model_save_path', type=str, default='checkpoint')
    parser.add_argument('--result_save_path', type=str, default='results')
    parser.add_argument('--target_size', type=int, default=256)
    parser.add_argument('--mask_mode', type=int, default=0)
    parser.add_argument('--g_path', type=str, default="checkpoint/100000.pth")
    parser.add_argument('--d_path', type=str, default="checkpoint/100000.pth")
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--n_threads', type=int, default=6)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--gpu_id', type=str, default="0")
    args = parser.parse_args()
        
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    model = PRVSModel()
    if args.test:
        model.initialize_model(args.g_path, train = False)
        model.cuda()
        dataloader = DataLoader(Dataset(args.data_root, args.mask_root, args.mask_mode, args.target_size, mask_reverse = True))
        model.test(dataloader, args.result_save_path)
    else:
        model.initialize_model(args.g_path, args.d_path, train = True)
        model.cuda()
        dataloader = DataLoader(Dataset(args.data_root, args.mask_root, args.mask_mode, args.target_size, mask_reverse = True), batch_size = args.batch_size, shuffle = True, num_workers = args.n_threads)
        model.train(dataloader, args.model_save_path, args.finetune)

if __name__ == '__main__':
    run()