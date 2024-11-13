import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lr = 0.001
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None

    def set_lr(self, lr):
        self.lr = lr
        if self.optimizer is not None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        pass

    def test_model(self, test_loader, is_show_info=True):
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            if is_show_info:
                print(f'Test accuracy: {(correct / total) * 100} %')
            return correct / total

    def train_model(self, train_loader, test_loader, num_epochs):
        res_output_step = 100
        train_loader.device = self.device
        total_step = len(train_loader)
        test_acc_list = []
        train_acc_list = []

        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self(images)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total = labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()

                if (i + 1) % res_output_step == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                          .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), (correct / total) * 100))

            train_acc = self.test_model(train_loader, False)
            train_acc_list.append(train_acc)
            test_acc = self.test_model(test_loader, False)
            test_acc_list.append(test_acc)
        return train_acc_list, test_acc_list

    def save_model(self, model_path):
        torch.save(self.state_dict(), model_path + '.ckpt')
        input_shape = (1, 1, 28, 28)
        sample_input = torch.randn(input_shape).to(self.device)
        torch.onnx.export(
            self,  # The model to be exported
            sample_input,  # The sample input tensor
            model_path + ".onnx",  # The output file name
            export_params=True,  # Store the trained parameter weights inside the model file
            opset_version=17,  # The ONNX version to export the model to
            do_constant_folding=True,  # Whether to execute constant folding for optimization
            input_names=['input'],  # The model's input names
            output_names=['output'],  # The model's output names
        )

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path, weights_only=False))
