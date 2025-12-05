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
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint

# ==========================================================================
# Define hyperparameters
# ==========================================================================
num_bags_to_display = 10  # number of bags to display results for during testing


# ==========================================================================
# Define functions
# ==========================================================================
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
    parser.add_argument(
        "--no_validation",
        action="store_true",
        default=False,
        help="do validation or not",
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
    # * data loader of validation set
    val_loader = data_utils.DataLoader(
        MnistBags(
            target_number=args.target_number,
            mean_bag_length=args.mean_bag_length,
            var_bag_length=args.var_bag_length,
            num_bag=args.num_bags_train,
            seed=args.seed,
            train=True,
        ),
        batch_size=1,
        shuffle=False,
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
        model = LitAttention(no_validation=args.no_validation)
    elif args.model == "gated_attention":
        model = LitGatedAttention(no_validation=args.no_validation)
    else:
        raise ValueError(
            f"Unknown model type: {args.model}. Choose 'attention' or 'gated_attention'"
        )

    # train the model
    # ---------------
    print("Start Training")
    if args.no_validation:
        print("ℹ️ No validation during training")
        callbacks = ModelCheckpoint(
            dirpath="./ckpt", monitor="train_loss", filename="admil"
        )
    else:
        print("ℹ️ Validation during training")
        callbacks = ModelCheckpoint(
            dirpath="./ckpt", monitor="val_loss", filename="admil"
        )

    trainer = pl.Trainer(
        devices="auto",
        accelerator="gpu" if args.cuda else "cpu",
        strategy="auto",
        max_epochs=args.epochs,
        callbacks=callbacks,
        fast_dev_run=False,
    )

    trainer.fit(model, train_loader, val_loader)

    # test the model
    # --------------
    print("Start Testing")
    trainer.test(model, test_loader)

    # model prediction
    # ----------------
    print("Start Predicting")
    prediction = trainer.predict(model, test_loader)
    if prediction is not None:
        print(f"Total number of bags predicted: {len(prediction)}")
        print(f"Displaying results for first {num_bags_to_display} bags:")
        for i in range(min(num_bags_to_display, len(prediction))):
            bag_level, instance_level = prediction[i]
            print(f"\nBag {i+1}:")
            print(f"  Bag-level (True Label, Predicted Label): {bag_level}")
            print("  Instance-level (True Label, Attention Weight):")
            for inst in instance_level:
                print(f"    {inst}")

    # [EOF]
