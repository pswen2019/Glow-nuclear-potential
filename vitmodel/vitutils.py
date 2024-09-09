import torch 

def loadvitmodel(model, path, device = None):
    if device is not None:
        checkpoint = torch.load(path, map_location = device)
    else:
        checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    if device is not None:
        model.to(device)
    return model

def loadmodelmethod(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"]())
    return model
