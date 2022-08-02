# using https://github.com/faustomorales/vit-keras library
# pip install vit-keras
from vit_keras import vit, utils


def VIT():
    image_size = 224

    model = vit.vit_l32(
        image_size=image_size,
        activation="sigmoid",
        pretrained=True,
        include_top=True,
        pretrained_top=False,
        classes=101,
    )

    return model


if __name__ == "__main__":
    model = VIT()
    model.summary()
