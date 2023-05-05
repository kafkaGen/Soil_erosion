import torch


class Config():
    # Training
    seed = 13
    epochs = 70
    learning_rate = 3e-4
    num_classes = 1
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    
    # Data preparation
    data_path = 'data/'
    model_path = 'models/'
    logs_path = 'tb_logs'
    batch_size = 16
    num_workers = 4
    
    # Data transformation
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    resize_to = 128