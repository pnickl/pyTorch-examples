import torch
import torch.onnx as onnx
import torchvision.models as models

# load pretrained model and save
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'tut_model_weights.pth')

# load untrained model and load saved params
model = model.vgg16()
model.load_state_dict(torch.load('tut_model_weights.pth'))
model.eval()    # this sets dropout and batch normalization layers to evaluation mode


# this is saving the whole model, including shape specification
torch.save(model, 'model.pth')
model = torch.load('model.pth')

# ONNX model
input_image = torch.zeros((1,3,224,224))
onnx.export(model, input_image, 'model.onnx')