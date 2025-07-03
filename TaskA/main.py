import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import argparse

from utils.data_processing import get_transforms, get_datasets_and_loaders, train_dir, val_dir,test_dir
from utils.model import get_model
from utils.train import train_model
from utils.test import test_model
from utils.plots import plots
from utils.evaluation import evaluate_model

def main(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    is_training_weight_available = True
    
    train_transform, val_transform, test_transform = get_transforms()
    
    train_loader, val_loader, train_dataset, val_dataset ,test_loader,test_dataset = get_datasets_and_loaders(
        train_transform, val_transform,test_transform,train_data_dir=args.train_data_dir,val_data_dir=args.val_data_dir,test_data_dir=args.test_data_dir, batch_size=args.batch_size
    ) # uncomment when using test dataset

    
    model = get_model(num_classes=2)
    model.to(device)

    if os.path.exists(args.model_save_path) and args.evaluate_only:
        print(f"Loading model weights from {args.model_save_path}.")
        model.load_state_dict(torch.load(args.model_save_path, map_location=device))
        is_training_weight_available=True
    elif args.evaluate_only:
        print(f"No existing model weights found at {args.model_save_path}. Training is required.")   
        is_training_weight_available = False
    else:
        print()

    
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay_factor, patience=args.lr_patience)

    if not args.evaluate_only:
        print("Starting training...")
        history = train_model(
            model, train_loader, val_loader, device,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=args.epochs,
            patience=args.early_stopping_patience,
            max_grad_norm=args.max_grad_norm,
            label_smoothing=args.label_smoothing,
            use_amp=args.use_amp,
            save_path=args.model_save_path
        )
        print("Training finished.")
        
         # Plotting
        if history:
            plots(history, save_dir=args.plot_save_dir)
            print("Training curves plotted and saved.")
        results = evaluate_model(model, val_loader, device,type='train',class_names=val_dataset.classes,save_dir=args.plot_save_dir)
        
    else:
        history = None

    # Testing
    if is_training_weight_available:
        print("\nStarting testing...")
        test_loss, test_acc = test_model(
            model, test_loader, device,
            criterion=criterion,
            save_path=args.model_save_path # Use the same path where best model was saved
        )
        print(f"Final Test Loss: {test_loss:.4f} | Final Test Accuracy: {test_acc:.4f}")
        results = evaluate_model(model, test_loader, device,type='test',class_names=test_dataset.classes,save_dir=args.plot_save_dir)
        # print(results)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gender Classification Training Script')

    # Data arguments
    parser.add_argument('--train_data_dir', type=str, default=train_dir,
                        help='Path to the training data directory.')
    parser.add_argument('--val_data_dir', type=str, default=val_dir,
                        help='Path to the validation data directory.')
    parser.add_argument('--test_data_dir', type=str, default=test_dir, 
                        required=True, help='Path to the testing data directory.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for data loaders.')

    # Model arguments
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of output classes')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=3*1e-5,
                        help='Initial learning rate for the optimizer.')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 penalty) for the optimizer.')
    parser.add_argument('--label_smoothing', type=float, default=0.08,
                        help='Label smoothing factor for the loss function.')
    parser.add_argument('--use_amp', type=bool, default=True,
                        help='Whether to use Automatic Mixed Precision (AMP).')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for gradient clipping. Set to 0 for no clipping.')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                        help='Number of epochs to wait for validation loss improvement before stopping.')
    parser.add_argument('--lr_patience', type=int, default=3,
                        help='Patience for ReduceLROnPlateau scheduler.')
    parser.add_argument('--lr_decay_factor', type=float, default=0.5,
                        help='Factor by which the learning rate will be reduced.')
    
    # Testing 
    parser.add_argument('--evaluate_only', action='store_true',
                    help='If set, skips training and only performs evaluation on the loaded model.')

    # Output paths
    parser.add_argument('--model_save_path', type=str, default='best_model.pth',
                        help='Path to save the best trained model weights.')
    parser.add_argument('--plot_save_dir', type=str, default='plots',
                        help='Directory to save training plots.')

    args = parser.parse_args()


    main(args)

