import numpy as np 
import os 
import torch 

from torch.utils.data import Dataset

LABEL_EXTENSIONS = [".txt"]
DATA_EXTENSIONS = [".dat"]

hbarc = 197.3

def _is_pot(fname, data_extension):
    fname, ext = os.path.splitext(fname)
    return (ext.lower() in data_extension)

def _is_label(fname, label_extension):
    fname, ext = os.path.splitext(fname)
    return (ext.lower() in label_extension)

def _find_pot_and_label(data_dir, label_extension):
    data = []
    for root, _, fnames in sorted(os.walk(data_dir)):
        for fname in sorted(fnames):
            if _is_label(fname, label_extension):
                labelfilename = os.path.join(root, fname)
                fname_and_attr = np.genfromtxt(labelfilename, dtype = "str")
                alldatafname = fname_and_attr[:, 0]
                allattr = fname_and_attr[:, 1:]
                allattr = np.asarray(allattr, dtype = float)
                nsample, nattr = fname_and_attr.shape
                for isample in range(nsample):

                    datafname = alldatafname[isample]
                    attr = allattr[isample]

                    path = os.path.join(root, datafname)
                    attr = torch.from_numpy(attr).to(dtype = torch.float32)
                    attr[-1] = (attr[-1] - 400.)/(600.-400.)
                    data.append({
                        "path":  path, 
                        "y_onehot": attr
                        })
    return data

def loaddata(path):
    data = np.loadtxt(path)
    _, ncol = data.shape
    data = np.reshape(data, (-1, ncol, ncol))
    data = data*hbarc*hbarc
    data = np.float32(data)
    return data

class ChiralDataset(Dataset):
    def __init__(self, root_dir, 
                 LABEL_EXTENSION = LABEL_EXTENSIONS, 
                 loaddata = loaddata):
        super().__init__()
        self.data = _find_pot_and_label(root_dir, LABEL_EXTENSION)
        self.loaddata = loaddata

    def __getitem__(self, index):
        data = self.data[index]
        path = data["path"]
        attr = data["y_onehot"]
        image = self.loaddata(path)
        return {
                "x": image, 
                "y_onehot": attr
                }

    def __len__(self):
        return len(self.data)
