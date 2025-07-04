# main.py
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import argparse
import torch
import numpy as np
import random
from utils.data_processing import get_transforms,get_datasets_and_loaders,train_dir,val_dir,test_dir
from utils.model import SiameseNetwork, device
from utils.loss import SiameseHybridLoss
from utils.train import train_siamese_model, evaluate_siamese_model, plot_training_curves
from utils.test import test_model


def main(args):

    seed = 42

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  

    random.seed(seed)
    np.random.seed(seed)
    # Create data loaders
    is_training_weight_available = True
    train_transform, val_transform, test_transform = get_transforms()

    train_loader, val_loader,test_loader = get_datasets_and_loaders(
        train_transform, val_transform,test_transform,
        train_data_dir=args.train_data_dir,val_data_dir=args.val_data_dir,
        test_data_dir=args.test_data_dir
    )

    # Initialize model, criterion, optimizer, and scheduler
    model = SiameseNetwork(embedding_dim=args.embedding_dim)
    criterion = SiameseHybridLoss(margin=args.margin, alpha=args.alpha)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)



    if os.path.exists(args.model_save_path) and args.evaluate_only:
        print(f"Loading best model from {args.model_save_path} for final evaluation.")
        model.load_state_dict(torch.load(args.model_save_path))
    elif args.evaluate_only:
        print(f"No existing model weights found at {args.model_save_path}. Training is required.")   
        is_training_weight_available = False
    else:
        print()
    
    if not args.evaluate_only:
        # Train the model
        print("\nStarting model training...")
        history = train_siamese_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=args.epochs,
            patience=args.patience,
            use_amp=args.use_amp,
            max_grad_norm=args.max_grad_norm,
            cosine_threshold=args.cosine_threshold,
            save_path=args.model_save_path
        )
        print("Training finished.")

        if history:
         # Plot training curves
            print("\nPlotting training curves...")
            plot_training_curves(history,save_dir=args.plot_save_dir)

        results = evaluate_siamese_model(model, val_loader,threshold=args.cosine_threshold,plot_cm=True,save_dir=args.plot_save_dir)
    else:
        history = None


    if is_training_weight_available:
        print("\nStarting testing...")
        accuracy, _, _, f1 = test_model(
            model, test_loader, device,
            save_path=args.model_save_path # Use the same path where best model was saved
        )
        print(f"Top-1 Accuracy: {accuracy:.4f} | F1-Score : {f1:.4f}")
        results = evaluate_siamese_model(
        model,
        test_loader,
        threshold=args.cosine_threshold,
        plot_cm=True,
        save_dir=args.plot_save_dir
         )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Recognition Task")
    
    parser.add_argument('--train_data_dir', type=str,default=train_dir,
                        help="Path to the training data root directory.")
    parser.add_argument('--val_data_dir', type=str,default=val_dir,
                        help="Path to the validation data root directory.")
    parser.add_argument('--test_data_dir', type=str, default=test_dir, required=True,
                        help="Path to the validation data root directory.")
    parser.add_argument('--evaluate_only', action='store_true',
                    help='If set, skips training and only performs evaluation on the loaded model.')

    parser.add_argument('--epochs', type=int, default=50,
                        help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size for data loaders.")
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help="Dimension of the learned embeddings.")
    parser.add_argument('--learning_rate', type=float, default=1e-6,
                        help="Initial learning rate for the optimizer.")
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help="Weight decay for the optimizer.")
    parser.add_argument('--margin', type=float, default=1.0,
                        help="Margin for the contrastive loss component.")
    parser.add_argument('--alpha', type=float, default=0.5,
                        help="Weighting factor for the hybrid loss (alpha * contrastive + (1-alpha) * cosine_bce).")
    parser.add_argument('--patience', type=int, default=5,
                        help="Patience for early stopping.")
    parser.add_argument('--use_amp', type=bool, default=True,
                        help="Whether to use Automatic Mixed Precision (AMP).")
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help="Maximum gradient norm for clipping.")
    parser.add_argument('--cosine_threshold', type=float, default=0.5,
                        help="Threshold for cosine similarity to classify 'same' during evaluation.")
    
    parser.add_argument('--model_save_path', type=str, default="best_model.pth",
                        help="Name of the file to save the best model.")
    parser.add_argument('--plot_save_dir', type=str, default='plots',
                        help='Directory to save training plots.')
    args = parser.parse_args()

    main(args)
