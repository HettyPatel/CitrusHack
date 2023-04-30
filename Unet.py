import torch

class UNET(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNET, self).__init__()
        self.conv1_1 = torch.nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv1_2 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.conv2_1 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = torch.nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_1 = torch.nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = torch.nn.Conv2d(256, 256, 3, padding=1)
        
        self.unpool2 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv2_1 = torch.nn.Conv2d(256, 128, 3, padding=1)
        self.upconv2_2 = torch.nn.Conv2d(128, 128, 3, padding=1)
        self.unpool1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv1_1 = torch.nn.Conv2d(128, 64, 3, padding=1)
        self.upconv1_2 = torch.nn.Conv2d(64, 64, 3, padding=1)
        
        
        self.out = torch.nn.Conv2d(64, out_channels, kernel_size=1, padding=0)
        
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.1)
        
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
    
    
    def forward(self, x):

        conv1 = self.relu(self.conv1_2(self.relu(self.conv1_1(x))))
        maxpool1 = self.maxpool(conv1)
        conv2 = self.relu(self.conv2_2(self.relu(self.conv2_1(maxpool1))))
        maxpool2 = self.maxpool(conv2)
        conv3 = self.relu(self.conv3_2(self.relu(self.conv3_1(maxpool2))))

        unpool2 = self.unpool2(conv3)
        upconv2 = self.relu(self.upconv2_2(self.relu(self.upconv2_1((conv2, unpool2)))))
        unpool1 = self.unpool1(upconv2)
        upconv1 = self.relu(self.upconv1_2(self.relu(self.upconv1_1((conv1, unpool1)))))

        out = self.out(upconv1)
        return out
