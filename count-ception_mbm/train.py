import argparse
import torch

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.dataset import MBM
from model import ModelCountception
import numpy as np


parser = argparse.ArgumentParser(description='PyTorch Sealion count training')
parser.add_argument('--pkl-file', default="utils/MBM-dataset.pkl", type=str, help='path to pickle file.')
parser.add_argument('--batch-size', default=2, type=int, help='the batch size for training.')
parser.add_argument('--epochs', default=1000, type=int, help='total number of training epochs.')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate.')


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_dataset = MBM(pkl_file='train-dataset.pkl', transform=transforms.Compose([transforms.ToTensor()]), mode='full')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    val_dataset = MBM(pkl_file='valid-dataset.pkl', transform=transforms.Compose([transforms.ToTensor()]), mode='full')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    criterion = nn.L1Loss()
    model = ModelCountception().to(device)
    solver = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        for idx, (input, target, target_count) in enumerate(train_dataloader):
            input = input.to(device)
            target = target.to(device)
            output = model.forward(input)

            patch_size = 32
            ef = ((patch_size / 1) ** 2.0)
            output_count = (output.cpu().detach().numpy() / ef).sum(axis=(2, 3))
            target_count = target_count.data.cpu().detach().numpy()

            print(f"Output: {output_count} - Target: {target_count}")
            loss = criterion(output, target)

            # Zero grad
            model.zero_grad()
            loss.backward()
            solver.step()

        with torch.no_grad():
            val_loss = []
            for idx, (input, target, _) in enumerate(val_dataloader):
                input = input.to(device)
                target = target.to(device)
                output = model.forward(input)
                val_loss.append(criterion(output, target).data.cpu().numpy())
            print("Epoch", epoch, "- Validation Loss:", np.mean(val_loss))

        if (epoch+1) % 50 == 0:
            state = {'model_weights': model.state_dict()}
            torch.save(state, "checkpoints/after_{0}_epochs.model".format(epoch))


if __name__ == '__main__':
    main()
