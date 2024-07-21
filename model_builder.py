import torch

class FSRCNN(torch.nn.Module):
    def __init__(self, d, s, m):
        super().__init__()
        self.feature_extraction = torch.nn.Sequential(torch.nn.Conv2d(kernel_size=5, in_channels=3, out_channels=d, padding=2),
                                                      torch.nn.PReLU(d))

        self.shrinking = torch.nn.Sequential(torch.nn.Conv2d(kernel_size=1, in_channels=d, out_channels=s),
                                             torch.nn.PReLU(s))

        self.non_linear_mapping = []
        for _ in range(m):
            self.non_linear_mapping.extend([torch.nn.Conv2d(kernel_size=3, in_channels=s, out_channels=s, padding=1),torch.nn.PReLU(s)])
        self.non_linear_mapping = torch.nn.Sequential(*self.non_linear_mapping)

        self.expanding = torch.nn.Sequential(torch.nn.Conv2d(kernel_size=1, in_channels=s, out_channels=d),
                                             torch.nn.PReLU(d))

        self.deconvolution = torch.nn.ConvTranspose2d(kernel_size=9, in_channels=d, out_channels=3, stride=4, padding=3, output_padding=0)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in')
            elif isinstance(m, torch.nn.ConvTranspose2d):
                torch.nn.init.normal_(m.weight, mean=0, std=0.001)

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.shrinking(x)
        x = self.non_linear_mapping(x)
        x = self.expanding(x)
        x = self.deconvolution(x)
        return x
