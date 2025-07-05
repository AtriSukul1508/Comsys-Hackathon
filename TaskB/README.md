# Face Recognition (Task B)


## Directory Structure
```
TaskB/
├── model/
│   ├── TaskB.png    
├── plots/
│   ├── train_confusion_matrix.png
│   ├── training_curves.png
│   ├── val_confusion_matrix.png  
├── results/
│   ├── results.txt    
│   ├── train_confusion_matrix.png
│   ├── training_curves.png
│   ├── val_confusion_matrix.png  
├── utils/
│   ├── data_processing.py     
│   ├── loss.py                 
│   ├── model.py          
│   ├── test.py  
│   └── train.py             
├── main.py  
├── best_model.pth                   
├── requirements.txt           
└── README.md                 
```

Setup

Install dependencies from requirements.txt:

```pip install -r requirements.txt```

Data Preparation
The code expects your dataset to be organized in the following structure:
```
data/
├── Task_B/
        ├── train/
        │   ├── class1/
        │   │   ├── image.jpg
        │   │   └── ...
        │   └── classN/
        │       ├── image.jpg
        │       └── ...
        └── val/
            ├── class1/
            │   ├── image.jpg
            │   └── ...
            └── classN/
                ├── image.jpg
                └── ...
```
By default, the ```utils/data_processing.py``` script uses ```train_dir,val_dir``` and ```test_dir``` variables values as directory. We can change these paths via command-line arguments in ```main.py``` to point to our actual data locations.

## How to Run
The ```main.py``` script handles the entire workflow. We can run it from the project's root directory (**TaskB/**) using the ```python main.py``` command, along with various arguments to control its behavior.

```--evaluate_only```: Performs testing only. This requires either a pre-trained model to be available at the path specified by ```--model_save_path``` or model training. By default ```--model_save_path``` value is set to ```best_model.pth```.

To evaluate a model on a separate test set: we need another additional argument ```--test_data_dir``` along with the ```--evaluate_only``` argument.

#### 
> [!IMPORTANT]
> 1. the ```--test_data_dir``` argument is required always. <br/>
> 2. The pre-trained model weights can be accessed and downloaded from [*best_model.pth*](https://drive.google.com/file/d/1mB9Lqozewq4QgigvqeLhURdgIKyrKcZD/view?usp=sharing). <br/>
> 3. To test using pre-trained model weights, run the command ```python main.py --evaluate_only --test_data_dir /path/to/test/dataset```, or alternatively, use ```python main.py --evaluate_only --test_data_dir /pata/to/test/dataset --model_save_path /path/to/model``` with the model_save_path specified. For performing both training and testing together, simply run ```python main.py --test_data_dir /path/to/test/dataset```.

## Command-line Arguments
Here's a detailed list of all available command-line arguments for ```main.py```:

> [!IMPORTANT]
> *The models are run with the default values of the arguments listed below.*

```--train_data_dir (type: str, default: '../data/Task_A/train')```: Path to the training data directory.

```--val_data_dir (type: str, default: '../data/Task_A/val')```: Path to the validation data directory.

```--test_data_dir (type: str, default: "",required=True)```: Path to the test data directory. Since, the test data is not given, default value of ```---test_data_dir``` is set to ```""```.

```--batch_size (type: int, default: 32)```: Batch size for data loaders.

```--num_classes (type: int, default: 2)```: Number of output classes for the model (e.g., 2 for male/female).

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

```--model_save_path (type: str, default: best_model.pth)```: Path to save the best trained model weights during training, or to load weights from for testing.

```--plot_save_dir (type: str, default: plots)```: Directory to save training plots (loss/accuracy curves) and the confusion matrix.

