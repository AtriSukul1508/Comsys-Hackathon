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

### Command-Line Arguments

```--train_data_dir (type: str, default: '../data/Task_B/train')```: Path to the training data root directory.

```--val_data_dir (type: str, default: '../data/Task_A/val')```: Path to the validation data root directory.

```--test_data_dir (type: str, default: "", required=True)```: Path to the test data root directory. Since, the test data is not given, default value of ```--test_data_dir``` is set to ```""```.

```--evaluate_only (action: store_true)```: If set, skips training and only performs evaluation on the loaded model.

```--epochs (type: int, default: 50)```: Number of training epochs.

```--batch_size (type: int, default: 16)```: Batch size for data loaders.

```--embedding_dim (type: int, default: 128)```: Dimension of the learned embeddings.

```--learning_rate (type: float, default: 1e-6)```: Initial learning rate for the optimizer.

```--weight_decay (type: float, default: 1e-4)```: Weight decay for the optimizer.

```--margin (type: float, default: 1.0)```: Margin for the contrastive loss component.

```--alpha (type: float, default: 0.5)```: Weighting factor for the hybrid loss (`alpha * contrastive + (1 - alpha) * cosine_bce`).

```--patience (type: int, default: 5)```: Patience for early stopping.

```--use_amp (type: bool, default: True)```: Whether to use Automatic Mixed Precision (AMP).

```--max_grad_norm (type: float, default: 1.0)```: Maximum gradient norm for clipping.

```--cosine_threshold (type: float, default: 0.5)```: Threshold for cosine similarity to classify "same" during evaluation.

#### Output Paths

```--model_save_path (type: str, default: "best_model.pth")```: Name of the file to save the best model.

```--plot_save_dir (type: str, default: "plots")```: Directory to save training plots.


