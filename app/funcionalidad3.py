import pickle
import torch 
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, in_features=9, h1=24, h2=32, out_features=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
    
with open('model_entire.pickle', 'rb') as f:
    loaded_model = pickle.load(f)

def funcionalidad3(respuestas_usuario):
  if loaded_model.forward(torch.FloatTensor(respuestas_usuario)).argmax().item() == 0:
     return "Por ahora no pareces estar en riesgo"
  return "Podrías estar en riesgo, te recomendamos realizarte una prueba de VIH lo más pronto posible"
     
#print(funcionalidad3([24, 3, 0, 0, 0, 0, 5, 1, 0]))