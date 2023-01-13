from src.project_config import project_config
from torch import nn
import torch

class PTRegression_config(project_config):
    def __init__(self,
                 encoder='flatten',
                 regresser_input=512,
                 regressor_output=1,
                 flatten_assist=False,
                 output_style='continuous',
                 **kwargs):
        super(PTRegression_config, self).__init__('model_PTRegression')
        self.encoder = encoder
        self.regresser_input = regresser_input
        self.flatten_assist = flatten_assist
        self.output_style = output_style
        if output_style!='continuous' and regressor_output==1:
            print('WARNING: PTRegression config: output style is not '
                  'continuous and your output size is 1. Output size must be '
                  '>1 to truely perform correctly. ')
        self.regressor_output = regressor_output



class PTRegressionModel(nn.Module):
    def __init__(self,
                 config=PTRegression_config(),
                 encoder=None):
        super(PTRegressionModel, self).__init__()
        self.config = config
        assert config.encoder=='custom' and encoder is not None
        if encoder is not None:
            self.encoder = encoder
        elif config.encoder=='flatten':
            self.encoder = nn.Sequential(
                nn.Flatten()
            )
        else:
            raise NotImplementedError(
                str(config.encoder) + ' is not an implemented in '
                                      'PTRegressionModel ')
        if self.config.flatten_assist:
            self.flattener = nn.Flatten()

        if self.config.output_style=='continuous':
            self.regresser = nn.Linear(
                config.regresser_input, self.config.regressor_output
            )
        elif self.config.output_style=='CORAL':
            assert self.config.regressor_output>1
            self.regresser = nn.Linear(self.config.regresser_input, 1, bias=False)
            self.coral_bias = nn.Parameter(
                0.5 * torch.ones(self.config.regressor_output-1).float())
        elif self.config.output_style=='softlabel' or self.config.output_style=='binary':
            assert self.config.regressor_output>1
            self.regresser = nn.Linear(self.config.regresser_input,
                                       self.config.regressor_output,
                                       bias=True)
        else:
            raise Exception('The regressor output_style is not recognized. '
                            + str(self.config.output_style))


    def forward(self, x, return_latent=False):
        latent = self.encoder(x)
        if self.config.flatten_assist:
            latent = self.flattener(latent)

        if latent.shape[1]!=self.regresser.in_features:
            raise Exception('The encoder and the regressor_input are mis-'
                            'matched.  The regressor input should be ' + str(
                latent.shape[1]) + ' and not ' + str(
                self.config.regresser_input))
        if self.config.output_style=='continuous' :
            return (self.regresser(latent), latent) if return_latent else self.regresser(latent)
        elif self.config.output_style=='CORAL':
            return (self.regresser(latent) + self.coral_bias,\
                   latent) if return_latent else self.regresser(latent) + self.coral_bias
        elif self.config.output_style=='softlabel' or self.config.output_style=='binary':
            return (self.regresser(latent), latent) if return_latent else self.regresser(latent)





