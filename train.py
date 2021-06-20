import os
import itertools

import torch
from absl import flags, app
from torchvision import transforms
from utils.dataset import BirdDataset
from utils.model import NetResDeep

FLAGS = flags.FLAGS

flags.DEFINE_enum('device', None, ['cpu', 'gpu'], 'Set which device should be used for training')

def main(argv):

    if FLAGS.device is None:
        if torch.cuda.is_available():
            device = 'gpu'
        else:
            device = 'cpu'
    else:
        device = FLAGS.device

    train = BirdDataset(train=True, transform=transforms.Compose([
        transforms.CenterCrop(416)
    ]))
    test = BirdDataset(train=False)
    imgs = torch.stack([img_t for img_t, _ in itertools.islice(train, 100)], dim=3)
 
    
    #model = NetResDeep(416, 32).to(device)
if __name__ == '__main__':
    app.run(main)
