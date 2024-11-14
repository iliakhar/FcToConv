import os.path
import sys
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
from FcConvNet import FcConvNet
from FullConvNet import FullConvNet


def create_plot(train_y: list, test_y: list, tittle=''):
    x = np.array([i for i in range(1, len(test_y)+1)])
    test_y = np.array(test_y)
    train_y = np.array(train_y)

    plt.plot(x, train_y, marker='o', markersize=4, label='test')
    plt.plot(x, test_y, marker='o', markersize=4, label='train')

    min_x, max_x, step_x = 0, len(x)+1, 1
    min_y, max_y, step_y = 0.9, 1.01, 0.01
    plt.xticks(np.arange(min_x, max_x, step_x))
    plt.yticks(np.arange(min_y, max_y, step_y))
    plt.grid()

    plt.title(tittle)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig('graph/' + tittle + '_graph.png', dpi=300, bbox_inches='tight')


def run_test(params: list, data_path, trans):
    number_of_test_params = 4
    if len(params) != number_of_test_params:
        raise Exception()
    conv_type, batch_size_str, model_path = params[1:]
    if conv_type == 'conv':
        model = FcConvNet()
    elif conv_type == 'full_conv':
        model = FullConvNet()
    else:
        raise Exception()

    batch_size = max(1, int(batch_size_str))

    if os.path.isfile(model_path):
        model.load_model(model_path)
        test_dataset = torchvision.datasets.MNIST(root=data_path, train=False, transform=trans)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        model.test_model(test_loader)
    else:
        raise Exception()


def run_train(params: list, data_path, trans):
    number_of_train_params = 7
    if len(params) != number_of_train_params:
        raise Exception()
    conv_type, num_epochs_str, lr_str, batch_size_str, folder, model_name = params[1:]
    if conv_type == 'conv':
        model = FcConvNet()
    elif conv_type == 'full_conv':
        model = FullConvNet()
    else:
        raise Exception()
    num_epochs = max(1, int(num_epochs_str))
    lr = float(lr_str)
    batch_size = max(1, int(batch_size_str))

    if not os.path.isdir(folder):
        raise Exception()

    train_dataset = torchvision.datasets.MNIST(root=data_path, train=True, transform=trans, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = torchvision.datasets.MNIST(root=data_path, train=False, transform=trans)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    model.set_lr(lr)
    train_acc_list, test_acc_list = model.train_model(train_loader, test_loader, num_epochs)
    model.save_model(folder+'/'+model_name)

    tittle = 'Convnet' if conv_type == 'conv' else 'Full_Convnet'
    # create_plot(train_acc_list, test_acc_list, tittle)


def main():
    args = sys.argv
    params = args[1:]

    norm_mean, norm_std = 0.1307, 0.3081
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((norm_mean,), (norm_std,))])
    data_path = 'dataset'
    try:
        if params[0] == 'test':
            run_test(params, data_path, trans)
        elif params[0] == 'train':
            run_train(params, data_path, trans)
    except Exception:
        print('Invalid input format')
        return


if __name__ == '__main__':
    main()
