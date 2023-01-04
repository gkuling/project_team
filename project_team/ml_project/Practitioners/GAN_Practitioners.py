from .PTRegression_Practitioner import PTRegression_Practitioner
import torch

class Discriminator_Practitioner(PTRegression_Practitioner):

    def calculate_loss(self, py, y):
        n = py.dim()-y.dim()
        y = torch.ones_like(py) * y[(None,)*n].permute(
            -2,-1,*[_ for _ in range(n)]
        )
        return self.loss_function(py, y)

