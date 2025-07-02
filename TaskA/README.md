# Gender Classification (Task A)


## Directory Structure
```
TaskA/
├── plots/
│   ├── confusion_matrix.png    
│   ├── training_curves.png  
├── utils/
│   ├── data_processing.py     
│   ├── evaluation.py                 
│   ├── model.py               
│   ├── plots.py        
│   ├── test.py  
│   └── train.py             
├── main.py  
├── model.pth                   
├── requirements.txt           
└── README.md                 
```

Setup
Clone the repository (or create the files manually):


Install dependencies from requirements.txt:

```pip install -r requirements.txt```

Data Preparation
The code expects your dataset to be organized in the following structure:
```
data/
├── train/
│   ├── male/
│   │   ├── image.jpg
│   │   └── ...
│   └── female/
│       ├── image.jpg
│       └── ...
└── val/
    ├── male/
    │   ├── image.jpg
    │   └── ...
    └── female/
        ├── image.jpg
        └── ...
```
By default, the ```utils/data_processing.py``` script uses ```train_dir,val_dir``` and ```test_dir``` variables values as directory. We can change these paths via command-line arguments in ```main.py``` to point to our actual data locations.

## How to Run
The ```main.py``` script orchestrates the entire workflow. We can run it from the project's root directory (**TaskA/**) using the ```python main.py``` command, along with various arguments to control its behavior.


```--evaluate_only```: Performs testing only. This mode requires a pre-trained model to be available at the path specified by ```--model_save_path```. By default ```--model_save_path``` value is set to ```model.pth```. Pretrained we


To evaluate a pre-trained model on a separate test set:
(Requires best_model.pth and your test data to be in the specified directory.)

```python main.py --evaluate_only --test_data_dir /path/to/your/dataset/test```

## Command-line Arguments
Here's a detailed list of all available command-line arguments for ```main.py```:

*The models are run with the default values of the arguments listed below unless explicitly overridden via command-line arguments.*

Data Arguments:

```--train_data_dir (type: str, default: '../data/Task_A/train')```: Path to the training data directory.

```--val_data_dir (type: str, default: '../data/Task_A/val')```: Path to the validation data directory.

```--test_data_dir (type: str, default: "")```: Path to the test data directory. Since, the test data is not given, default value of ```---test_data_dir``` is set to ```""```.

```--batch_size (type: int, default: 32)```: Batch size for data loaders.

Model Arguments:

```--num_classes (type: int, default: 2)```: Number of output classes for the model (e.g., 2 for male/female).

Training Arguments:

```--epochs (type: int, default: 50)```: Number of training epochs.

```--learning_rate (type: float, default: 3e-5)```: Initial learning rate for the optimizer.

```--weight_decay (type: float, default: 1e-4)```: Weight decay (L2 penalty) for the optimizer, helps prevent overfitting.

```--label_smoothing (type: float, default: 0.08)```: Label smoothing factor for the loss function, can improve generalization.

```--use_amp (action: store_true)```: If set, enables Automatic Mixed Precision (AMP) for faster training and reduced memory usage.

```--max_grad_norm (type: float, default: 1.0)```: Maximum gradient norm for gradient clipping. Set to 0 for no clipping. Helps prevent exploding gradients.

```--early_stopping_patience (type: int, default: 5)```: Number of epochs to wait for validation loss improvement before stopping training.

```--lr_patience (type: int, default: 3)```: Patience for ReduceLROnPlateau scheduler. Number of epochs with no improvement after which learning rate will be reduced.

```--lr_decay_factor (type: float, default: 0.5)```: Factor by which the learning rate will be reduced (e.g., new_lr = old_lr * factor).

Output Paths:


```--model_save_path (type: str, default: model.pth)```: Path to save the best trained model weights during training, or to load weights from for testing.

```--plot_save_dir (type: str, default: plots)```: Directory to save training plots (loss/accuracy curves) and the confusion matrix.

