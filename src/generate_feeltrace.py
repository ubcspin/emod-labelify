from utils import create_dataset
import os

if __name__ == '__main__':
    create_dataset(os.path.join('..', 'participant'), os.path.join('..', 'feeltrace'), num_workers=2)