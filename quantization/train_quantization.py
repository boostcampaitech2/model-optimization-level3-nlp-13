import datetime
import os
import time
import copy

import torch
import torch.utils.data
from torch import nn
import torchvision
import torch.quantization
import utils
from train_q import train_one_epoch, evaluate, load_data
import wandb

def main(args):
    

    # Set backend engine to ensure that quantized model runs on the correct kernels
    torch.backends.quantized.engine = args.backend
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True


    # Data loading code
    # print("Loading data")
    # train_dir = os.path.join(args.data_path, 'train')
    # val_dir = os.path.join(args.data_path, 'val')

    # dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)
    # data_loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=args.batch_size,
    #     sampler=train_sampler, num_workers=args.workers, pin_memory=True)

    # data_loader_test = torch.utils.data.DataLoader(
    #     dataset_test, batch_size=args.eval_batch_size,
    #     sampler=test_sampler, num_workers=args.workers, pin_memory=True)

    from src.dataloader import create_dataloader
    from src.utils.common import get_label_counts, read_yaml
    import yaml
    data_config = read_yaml(cfg=args.data)

    data_config["DATA_PATH"] = os.environ.get("SM_CHANNEL_TRAIN", data_config["DATA_PATH"])
    log_dir = os.environ.get("SM_MODEL_DIR", os.path.join("exp", args.savefolder_name))

    if os.path.exists(log_dir): 
        modified = datetime.datetime.fromtimestamp(os.path.getmtime(log_dir + '/best.pt'))
        new_log_dir = os.path.dirname(log_dir) + '/' + modified.strftime("%Y-%m-%d_%H-%M-%S")
        os.rename(log_dir, new_log_dir)
    os.makedirs(log_dir, exist_ok=True)

    if args.output_dir:
        utils.mkdir(args.output_dir)

    with open(os.path.join(log_dir, "data.yml"), "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)

    data_loader, data_loader_test, _ = create_dataloader(data_config)

    # when training quantized models, we always start from a pre-trained fp32 reference model
    model = torchvision.models.quantization.__dict__[args.model](pretrained=True, quantize=args.test_only)
    model.to(device)

    if not (args.test_only or args.post_training_quantize):
        model.fuse_model()
        model.qconfig = torch.quantization.get_default_qat_qconfig(args.backend)
        torch.quantization.prepare_qat(model, inplace=True)

        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=args.lr_step_size,
                                                       gamma=args.lr_gamma)

    # criterion = nn.CrossEntropyLoss()
    from src.loss import CustomCriterion

    criterion = CustomCriterion(
        samples_per_cls=get_label_counts(data_config["DATA_PATH"])
        if data_config["DATASET"] == "TACO"
        else None,
        device=device,
    )


    model_without_ddp = model

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        evaluate(model, criterion, data_loader_test, device=device)
        return

    model.apply(torch.quantization.enable_observer)
    model.apply(torch.quantization.enable_fake_quant)
    
    from src.trainer import TorchTrainer
    
    model_path = os.path.join(log_dir, "best.pt")

    trainer = TorchTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        scaler=False, #fp16
        device=device,
        model_save_path=model_path,
        verbose=1,
    )

    def _get_len_label_from_dataset(dataset) -> int:
        """Get length of label from dataset.

        Args:
            dataset: torch dataset

        Returns:
            A number of label in set.
        """
        if isinstance(dataset, torchvision.datasets.ImageFolder) or isinstance(
            dataset, torchvision.datasets.vision.VisionDataset
        ):
            return len(dataset.classes)
        elif isinstance(dataset, torch.utils.data.Subset):
            return _get_len_label_from_dataset(dataset.dataset)
        else:
            raise NotImplementedError

    num_classes = _get_len_label_from_dataset(data_loader.dataset)
    label_list = [i for i in range(num_classes)]

    for epoch in range(args.start_epoch, args.epochs):
       
        print('Starting training for epoch', epoch)

        # train_one_epoch(model, criterion, optimizer, data_loader, device, epoch,
        #                 args.print_freq)

        train_acc, train_f1 = trainer.train_one_epoch(data_loader, label_list)

        lr_scheduler.step()
        with torch.no_grad():
            if epoch >= args.num_observer_update_epochs:
                print('Disabling observer for subseq epochs, epoch = ', epoch)
                model.apply(torch.quantization.disable_observer)
            if epoch >= args.num_batch_norm_update_epochs:
                print('Freezing BN for subseq epochs, epoch = ', epoch)
                model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
            print('Evaluate QAT model')

            _, test_f1, test_acc = trainer.eval(
                        model=model, test_dataloader=data_loader_test, device =device
                    )

            # evaluate(model, criterion, data_loader_test, device=device)
            quantized_eval_model = copy.deepcopy(model_without_ddp)
            quantized_eval_model.eval()
            quantized_eval_model.to(torch.device('cpu'))
            torch.quantization.convert(quantized_eval_model, inplace=True)

            print('Evaluate Quantized model')
            # evaluate(quantized_eval_model, criterion, data_loader_test,
            #          device=torch.device('cpu'))

            _, test_f1_q, test_acc_q = trainer.eval(
                        model=quantized_eval_model, test_dataloader=data_loader_test, device =torch.device('cpu')
                    )

            wandb.log({
                        "train acc" : train_acc, 
                        "train f1": train_f1,
                        "validation acc" : test_acc, 
                        "validation f1": test_f1,
                        "validation_q acc" : test_acc_q, 
                        "validation_q f1": test_f1_q,
                        "epoch" : epoch
                    })

        model.train()

        if args.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'eval_model': quantized_eval_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args}
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'checkpoint.pth'))
        print('Saving models after epoch ', epoch)


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Quantized Classification Training', add_help=add_help)

    parser.add_argument('--data-path',
                        default='/datasets01/imagenet_full_size/061417/',
                        help='dataset')
    parser.add_argument('--model',
                        default='mobilenet_v2',
                        help='model')
    parser.add_argument('--backend',
                        default='qnnpack',
                        help='fbgemm or qnnpack')
    parser.add_argument('--device',
                        default='cuda',
                        help='device')

    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        help='batch size for calibration/training')
    parser.add_argument('--eval-batch-size', default=128, type=int,
                        help='batch size for evaluation')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--num-observer-update-epochs',
                        default=4, type=int, metavar='N',
                        help='number of total epochs to update observers')
    parser.add_argument('--num-batch-norm-update-epochs', default=3,
                        type=int, metavar='N',
                        help='number of total epochs to update batch norm stats')
    parser.add_argument('--num-calibration-batches',
                        default=32, type=int, metavar='N',
                        help='number of batches of training set for \
                              observer calibration ')

    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr',
                        default=0.0001, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum',
                        default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=30, type=int,
                        help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float,
                        help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=10, type=int,
                        help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. \
             It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--post-training-quantize",
        dest="post_training_quantize",
        help="Post training quantize the model",
        action="store_true",
    )

    parser.add_argument(
        "--data", default="configs/data/taco.yaml", type=str, help="data config"
    )
    
    parser.add_argument("--savefolder_name", type=str)
    
    
    parser.add_argument(
        "--output_dir", type=str)
    


    

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)