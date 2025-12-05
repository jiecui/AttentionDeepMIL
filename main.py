# Main script to train and test attention-based multiple instance learning models on MNIST bags

# 2025 Modified by Richard J. Cui. on Thu 12/04/2025 05:40:18.272526 PM
# $Revision: 0.2 $  $Date: Fri 12/05/2025 02:55:41.021060 PM $
#
# Mayo Clinic Foundation
# Rochester, MN 55901, USA
#
# Email: Cui.Jie@mayo.edu

# ==========================================================================
# Import libraries
# ==========================================================================
import os
import argparse
import torch
import torch.optim as optim
import torch.utils.data as data_utils
import pytorch_lightning as pl
from dataloader import MnistBags
from model import LitAttention, LitGatedAttention
from pytorch_lightning.callbacks import ModelCheckpoint

# ==========================================================================
# Define hyperparameters
# ==========================================================================
num_bags_to_display = 5  # number of bags to display results for during testing


# ==========================================================================
# Define functions
# ==========================================================================
# def train(
#     model: Attention | GatedAttention, optimizer: torch.optim.Optimizer, epoch: int
# ):
#     model.train()
#     train_loss = 0.0
#     train_error = 0.0
#     for data, label in train_loader:
#         bag_label = label[0]
#         if args.cuda:
#             data, bag_label = data.cuda(), bag_label.cuda()
#         # data, bag_label = Variable(data), Variable(bag_label)

#         # reset gradients
#         optimizer.zero_grad()
#         # calculate loss and metrics
#         loss, _ = model.calculate_objective(data, bag_label)
#         train_loss += loss.data[0].item()
#         error, _ = model.calculate_classification_error(data, bag_label)
#         train_error += error
#         # backward pass
#         loss.backward()
#         # step
#         optimizer.step()

#     # calculate loss and error for epoch
#     train_loss /= len(train_loader)
#     train_error /= len(train_loader)

#     print(
#         "Epoch: {}, Loss: {:.4f}, Train error: {:.4f}".format(
#             epoch, train_loss, train_error
#         )
#     )


# def test(model: Attention | GatedAttention, num_bags_to_display=5):
#     model.eval()
#     test_loss = 0.0
#     test_error = 0.0

#     with torch.no_grad():
#         for batch_idx, (data, label) in enumerate(test_loader):
#             bag_label = label[0]
#             instance_labels = label[1]
#             if args.cuda:
#                 data, bag_label = data.cuda(), bag_label.cuda()
#             data, bag_label = Variable(data), Variable(bag_label)
#             loss, attention_weights = model.calculate_objective(data, bag_label)
#             test_loss += loss.data[0].item()
#             error, predicted_label = model.calculate_classification_error(
#                 data, bag_label
#             )
#             test_error += error

#             if (
#                 batch_idx < num_bags_to_display
#             ):  # plot bag labels and instance labels for first 5 bags
#                 bag_level = (
#                     bag_label.cpu().data.numpy()[0],
#                     int(predicted_label.cpu().data.numpy()[0][0]),
#                 )
#                 instance_level = list(
#                     zip(
#                         instance_labels.numpy()[0].tolist(),
#                         np.round(
#                             attention_weights.cpu().data.numpy()[0], decimals=3
#                         ).tolist(),
#                     )
#                 )

#                 print(
#                     "\nTrue Bag Label, Predicted Bag Label: {}\n"
#                     "True Instance Labels, Attention Weights: {}".format(
#                         bag_level, instance_level
#                     )
#                 )

#     test_error /= len(test_loader)
#     test_loss /= len(test_loader)

#     print("\nTest Set, Loss: {:.4f}, Test error: {:.4f}".format(test_loss, test_error))

if __name__ == "__main__":
    # define command-line arguments
    # -----------------------------
    parser = argparse.ArgumentParser(description="PyTorch MNIST bags Example")
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        metavar="N",
        help="number of epochs to train (default: 20)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0005,
        metavar="LR",
        help="learning rate (default: 0.0005)",
    )
    parser.add_argument(
        "--reg", type=float, default=10e-5, metavar="R", help="weight decay"
    )
    parser.add_argument(
        "--target_number",
        type=int,
        default=9,
        metavar="T",
        help="bags have a positive labels if they contain at least one 9",
    )
    parser.add_argument(
        "--mean_bag_length",
        type=int,
        default=10,
        metavar="ML",
        help="average bag length",
    )
    parser.add_argument(
        "--var_bag_length",
        type=int,
        default=2,
        metavar="VL",
        help="variance of bag length",
    )
    parser.add_argument(
        "--num_bags_train",
        type=int,
        default=200,
        metavar="NTrain",
        help="number of bags in training set",
    )
    parser.add_argument(
        "--num_bags_test",
        type=int,
        default=50,
        metavar="NTest",
        help="number of bags in test set",
    )
    parser.add_argument(
        "--num_cpu_workers",
        type=int,
        default=1,
        metavar="NCPU",
        help="number of cpu workers. if < 1, set to all of available CPUs",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="attention",
        help="Choose b/w attention and gated_attention",
    )

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # set parameters
    # --------------
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        print("\nGPU is ON!")

    if args.num_cpu_workers > 0:
        num_workers = args.num_cpu_workers
    else:
        cpu_count = os.cpu_count()
        num_workers = max(1, cpu_count) if cpu_count else 1
    print(f"Using {num_workers} CPU workers for data loading\n")

    # setup datasets
    # --------------
    print("Setup training and testing datasets")
    loader_kwargs = {"num_workers": num_workers, "pin_memory": True}
    # * data loader of training set
    train_loader = data_utils.DataLoader(
        MnistBags(
            target_number=args.target_number,
            mean_bag_length=args.mean_bag_length,
            var_bag_length=args.var_bag_length,
            num_bag=args.num_bags_train,
            seed=args.seed,
            train=True,
        ),
        batch_size=1,
        shuffle=True,
        persistent_workers=True,
        **loader_kwargs,
    )
    # * data loader of testing set
    test_loader = data_utils.DataLoader(
        MnistBags(
            target_number=args.target_number,
            mean_bag_length=args.mean_bag_length,
            var_bag_length=args.var_bag_length,
            num_bag=args.num_bags_test,
            seed=args.seed,
            train=False,
        ),
        batch_size=1,
        shuffle=False,
        persistent_workers=True,
        **loader_kwargs,
    )

    # initialize model and optimizer
    # ------------------------------
    print("Init Model")
    if args.model == "attention":
        model = LitAttention()
    elif args.model == "gated_attention":
        model = LitGatedAttention()
    else:
        raise ValueError(
            f"Unknown model type: {args.model}. Choose 'attention' or 'gated_attention'"
        )

    # train the model
    # ---------------
    print("Start Training")
    callbacks = ModelCheckpoint(
        dirpath="./chpt", monitor="train_loss", filename="admil"
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if args.cuda else "cpu",
        devices=1 if args.cuda else 1,
        callbacks=callbacks,
        fast_dev_run=False,
    )
    trainer.fit(model, train_loader)

    # test the model
    # --------------
    print("Start Testing")
    # test(model, num_bags_to_display)
    trainer.test(model, test_loader)

    # [EOF]
