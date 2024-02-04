import torch
from torch import nn
from torchvision.transforms import v2
import gradio as gr


class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, apply_batchnorm=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        )
        if apply_batchnorm:
            self.conv.append(nn.BatchNorm2d(out_channels))
        self.conv.append(nn.LeakyReLU())

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, apply_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        )
        if apply_dropout:
            self.conv.append(nn.Dropout())
        self.conv.append(nn.ReLU())

    def forward(self, x):
        return self.conv(x)


class Generator(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder_blocks = nn.Sequential(
            EncoderBlock(in_channels, 64, apply_batchnorm=False),  # 128
            EncoderBlock(64, 128),  # 64
            EncoderBlock(128, 256),  # 32
            EncoderBlock(256, 512),  # 16
            EncoderBlock(512, 512),  # 8
            EncoderBlock(512, 512),  # 4
            EncoderBlock(512, 512),  # 2
            EncoderBlock(512, 512),  # 1
        )

        self.decoder_blocks = nn.Sequential(
            DecoderBlock(512, 512, apply_dropout=True),  # 2
            DecoderBlock(512 + 512, 512, apply_dropout=True),  # 4
            DecoderBlock(512 + 512, 512, apply_dropout=True),  # 8
            DecoderBlock(512 + 512, 512),  # 16
            DecoderBlock(512 + 512, 256),  # 32
            DecoderBlock(256 + 256, 128),  # 64
            DecoderBlock(128 + 128, 64),  # 128
        )

        self.out = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def encode(self, x):
        skip_list = []
        for block in self.encoder_blocks:
            x = block(x)
            skip_list.append(x)
        skip_list = reversed(skip_list[:-1])
        return x, skip_list

    def decode(self, x, skip_list):
        for block, skip in zip(self.decoder_blocks, skip_list):
            x = block(x)
            x = torch.cat([x, skip], dim=1)
        x = self.out(x)
        return x

    def forward(self, x):
        x, skip_list = self.encode(x)
        return self.decode(x, skip_list)


# У меня почему-то выдавало ошибку без определения лосса из ноутбука, поэтому ниже затычка
def generator_loss():
    pass


generator = Generator(in_channels=1, out_channels=1)

checkpoint = torch.load("generator_100.pt", map_location=torch.device('cpu'))
generator.load_state_dict(checkpoint['model_state_dict'])

generator.train()  # в архитектуре pix2pix на инференсе тоже train

transforms = v2.Compose([
    v2.ToTensor(),
    v2.Grayscale(num_output_channels=1),  # картинки серые
    v2.Normalize(mean=[0.5], std=[0.5])
])

title = "Генерация портрета по очертанию"


# в итоге лучше всех работает неправильная версия с батчом в слое 1 на 1)))
# поэтому приходится делать финт ушами, потому что нельзя подавать батч 1
def generate(inp):
    print(inp)
    inp = transforms(inp).unsqueeze(0)
    inp = torch.cat((inp, inp), dim=0)
    with torch.set_grad_enabled(False):
        return v2.functional.to_pil_image(generator(inp)[0] * 0.5 + 0.5)


gr.Interface(fn=generate,
             inputs=gr.Image(type="pil", height=256, width=256),
             outputs=gr.Image(type="pil", height=256, width=256),
             examples=[f"{i}.jpg" for i in range(1, 11)],
             title=title).launch()
