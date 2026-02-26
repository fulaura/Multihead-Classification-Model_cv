import torch
from torchview import draw_graph
from f4_model import *

model = torch.load(r'D:\Projects\code\01-learning\AITU\3rd course\Applied Machine Learning\final\inf_2\models\face_v2_a1.pth', map_location='cpu', weights_only=False)
graph = draw_graph(model, input_size=(1, 3, 224, 224), expand_nested=True)
graph.visual_graph.render("pytorch_visualized2", format='png')