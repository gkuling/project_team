from project_team import project_config
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
        '''
        Pyotorhc model with a regression head
        :param encoder: a name for the given encoder or "flatten" which will
        just flattent all dimensions after batchsize
        :param regresser_input: input size for regression head
        :param regressor_output: output size for regression head
        :param flatten_assist: assistance in flattening before sending it
        through the regression head. default: False
        :param output_style: set up to do other options for regression.
            'continuous': a single output number from a linear layer
            'CORAL': a vector of outputs for consistent rank ordinal regression
            'softlabel': a n length output where index of the value is the
            regressed amount
            'binary': a 2 length output where the likelihood of 1 is the
            regression amount.
        '''
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
        '''
        pytrochr egression model
        :param config: confiuguration from above
        :param encoder: model encoder
        '''
        super(PTRegressionModel, self).__init__()
        self.config = config

        # if a custom encoder is disclosed, it must be given
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
        # have a flattener if assistance is needed
        if self.config.flatten_assist:
            self.flattener = nn.Flatten()

        # determine the output style
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
        '''
        forward pass on x
        :param x: input tensor
        :param return_latent: bool. Whether to return the model latent tensor
        from before the regssion head
        :return: output of the model and laten space is requested
        '''

        # obtain the latent space
        latent = self.encoder(x)

        # flatten if needed
        if self.config.flatten_assist:
            latent = self.flattener(latent)

        # check that the input channels match the regressor head
        if latent.shape[1]!=self.regresser.in_features:
            raise Exception('The encoder and the regressor_input are mis-'
                            'matched.  The regressor input should be ' + str(
                latent.shape[1]) + ' and not ' + str(
                self.config.regresser_input))

        # run regression head in the varying forms
        if self.config.output_style=='continuous' :
            return (self.regresser(latent), latent) if return_latent \
                else self.regresser(latent)
        elif self.config.output_style=='CORAL':
            return (self.regresser(latent) + self.coral_bias, latent) if \
                return_latent \
                else self.regresser(latent) + self.coral_bias
        elif self.config.output_style=='softlabel' or \
                self.config.output_style=='binary':
            return (self.regresser(latent), latent) if return_latent \
                else self.regresser(latent)
