class DefaultConfigs(object):
    n_classes = 192  ## number of classes
    img_weight = 224  ## image width
    img_height = 224  ## image height
    batch_size = 18 ## batch size
    epochs = 8    ## epochs
    learning_rate=0.00045  ## learning rate
    model_name='resnet34'
    checkpoints = f"model_{model_name}_Ep_{epochs}_lr_{learning_rate}_bs_{batch_size}.pt"
    store_json = f"Loss_{model_name}_Ep_{epochs}_lr_{learning_rate}_bs_{batch_size}.json"
config = DefaultConfigs()
