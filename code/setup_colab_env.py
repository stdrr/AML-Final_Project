import argparse
import os
from google.colab import drive



def setup_colab_env(dataset=None):
    """
    """
    # Check GPU
    os.system('nvidia-smi')
    # Mount drive
    drive.mount('/gdrive')
    # # Copy the data into the VM
    # os.system(f'time cp -r /gdrive/MyDrive/University/Second_year/AML/AML-Final_Project/data/{dataset} /content/data/{dataset}')
    # Set current project directory
    os.chdir('/gdrive/MyDrive/University/Second_year/AML/AML-Final_Project')
    # Install dependencies
    os.system('pip install pillow==7.1.2')
    os.chdir('code/conditional-lane-detection/')
    os.system('pip install -r requirements/build.txt')
    os.system('time python setup.py develop')
    # Symbolic link to the data path
    try:
        os.system('unlink data')
    except:
        pass
    os.system('ln -s /gdrive/MyDrive/University/Second_year/AML/AML-Final_Project/data data')



def parse_args():
    """
    """
    parser = argparse.ArgumentParser(description='Setup Colab Environment')
    parser.add_argument('--dataset', type=str, help='Name of the folder (without /) of the dataset')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    setup_colab_env(args.dataset)