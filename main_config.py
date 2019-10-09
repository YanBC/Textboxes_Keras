# from models.conv_model import conv_model as get_model
# from models.densenet_model import densenet_model as get_model
from models.separable_densenet_model import densenet_model as get_model

if __name__ == '__main__':
    model = get_model(512, 512)

    model.summary(line_length=120)