import os
import sys
import numpy as np
import torch
import torchvision
import argparse
import yaml
from modules import transform, resnet, network, contrastive_loss
from utils import yaml_config_hook, save_model
from torch.utils import data
from datasets.wsi_dataset import WSIDataset
from datasets.bcss_dataset import BCSSDataset


def train():
    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(data_loader):
        optimizer.zero_grad()
        x_i = x_i.to('cuda')
        x_j = x_j.to('cuda')
        z_i, z_j = model(x_i, x_j)
        loss_instance = criterion_instance(z_i, z_j)
        loss = loss_instance
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(
                f"Step [{step}/{len(data_loader)}]\t loss_instance: {loss_instance.item()}")
        loss_epoch += loss.item()
    return loss_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument("--config", type=str, help="experiment config")
    # load from cmd
    given_configs, remaining = parser.parse_known_args()
    # load from config yaml
    with open(given_configs.config) as f:
        cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)
    args = parser.parse_args()

    # change arg type
    if hasattr(args, 'learning_rate'):
        args.learning_rate = float(args.learning_rate)
    if hasattr(args, 'weight_decay'):
        args.weight_decay = float(args.weight_decay)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # prepare data
    if args.dataset == "bcss":
        print("Using BCSS dataset with BCSSDataset class")
        # BCSS has 5 classes (0: background, 1-4: tissue types)
        class_num = 5
        
        # Create custom transform for contrastive learning
        train_transform = transform.Transforms(size=args.image_size, blur=True)
        
        # Wrap the dataset to return two augmented views
        class ContrastiveDataset(torch.utils.data.Dataset):
            def __init__(self, bcss_dataset, transform):
                self.bcss_dataset = bcss_dataset
                self.transform = transform
            
            def __len__(self):
                return len(self.bcss_dataset)
            
            def __getitem__(self, idx):
                image, label = self.bcss_dataset[idx]
                # Apply transform twice to get two augmented views
                x_i, x_j = self.transform(image)
                return (x_i, x_j), label
        
        # Create BCSS dataset without transform (will apply in ContrastiveDataset)
        bcss_base = BCSSDataset(
            csv_path=args.df_list,
            split='train',
            transform=None,  # No transform here
            return_mask=False,
            label_mode='dominant'  # Use dominant class for filtering
        )
        dataset = ContrastiveDataset(bcss_base, train_transform)
        
    elif args.dataset == "wsi":
        # Use original WSIDataset for other datasets
        dataset = WSIDataset(args.df_list, transform=transform.Transforms(size=args.image_size, blur=True))
        class_num = 7
        
    else:
        raise NotImplementedError
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )
    
    print(f"Dataset: {args.dataset}, Samples: {len(dataset)}, Batches: {len(data_loader)}, Classes: {class_num}")
    # initialize model
    res = resnet.get_resnet(args.resnet)
    model = network.Network(res, args.feature_dim, class_num)
    model = model.to('cuda')
    # optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.reload:
        model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.start_epoch))
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1
    loss_device = torch.device("cuda")
    criterion_instance = contrastive_loss.InstanceLoss(args.batch_size, args.instance_temperature, loss_device).to(
        loss_device)
    criterion_cluster = contrastive_loss.ClusterLoss(class_num, args.cluster_temperature, loss_device).to(loss_device)
    # train
    for epoch in range(args.start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train()
        if epoch % 50 == 0:
            save_model(args, model, optimizer, epoch)
        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)}")
    save_model(args, model, optimizer, args.epochs)
