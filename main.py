import os.path
import sys
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
from FcConvNet import FcConvNet
from FullConvNet import FullConvNet


def run_test(params: list, data_path, trans):
    if len(params) != 4:
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
    if len(params) != 6:
        raise Exception()
    conv_type, num_epochs_str, lr_str, batch_size_str, model_name = params[1:]
    if conv_type == 'conv':
        model = FcConvNet()
    elif conv_type == 'full_conv':
        model = FullConvNet()
    else:
        raise Exception()
    num_epochs = max(1, int(num_epochs_str))
    lr = float(lr_str)
    batch_size = max(1, int(batch_size_str))

    train_dataset = torchvision.datasets.MNIST(root=data_path, train=True, transform=trans, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    model.set_lr(lr)
    model.train_model(train_loader, num_epochs)
    model.save_model(model_name)


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
