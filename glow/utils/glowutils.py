import torch

def savemodel(graph, path, optim = None):
    if optim is None:
        state = {"model_state_dict": graph.state_dict()}
    else:
        state = {"model_state_dict": graph.state_dict(), 
                 'optimizer_state_dict': optim.state_dict()}
    torch.save(state, path)

def loadmodel(path, graph, optim = None, device = torch.device("cpu")):
    checkpoint = torch.load(path, map_location = device)
    graph.load_state_dict(checkpoint['model_state_dict'])
    if (optim is not None) and ("optimizer_state_dict" in checkpoint):
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
    return graph, optim

def sympot(pot):
    single_channel_id = [0, 1, 2, 3, 4, 5, 8, 9, 10, 13]
    mix_channel_id = [6, 7, 11, 12]
    pot[:, single_channel_id] = (
            pot[:, single_channel_id] + torch.transpose(
                pot[:, single_channel_id], dim0 = -1, dim1 = -2
                )
            )/2

    mix_channel_1 = torch.transpose(
            pot[:, mix_channel_id[0::2]], dim0 = -1, dim1 = -2
            )
    mix_channel_2 = pot[:, mix_channel_id[1::2]]
    mix_channel = (mix_channel_1 + mix_channel_2)/2.0
    pot[:, mix_channel_id[0::2]] = torch.transpose(
            mix_channel, dim0=-1, dim1=-2
            )
    pot[:, mix_channel_id[1::2]] = mix_channel
    return pot
