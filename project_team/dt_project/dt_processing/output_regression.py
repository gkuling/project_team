from . import _TensorProcessing
from scipy.special import softmax

class Ordinal_label(_TensorProcessing):
    def __init__(self, ordinal_label, field_oi='y'):
        super(Ordinal_label, self).__init__()
        self.ordinal_label = ordinal_label
        self.field_oi = field_oi

    def __call__(self, ipt):
        position = [i for i, lbl in enumerate(self.ordinal_label)
                    if ipt[self.field_oi] in lbl]
        if len(position)>1:
            print('Warning: CustomTensorProcessing.py line 267, too many '
                  'positional labels. ')
        position = position[0]
        scale = 0.5
        ipt[self.field_oi] = (position+scale) / len(self.ordinal_label)
        return ipt

class Binary_label(_TensorProcessing):
    def __init__(self, ordinal_label, field_oi='y'):
        super(Binary_label, self).__init__()
        assert len(ordinal_label)==2
        self.ordinal_label = ordinal_label
        self.field_oi = field_oi

    def __call__(self, ipt):
        position = [i for i, lbl in enumerate(self.ordinal_label)
                    if ipt[self.field_oi] in lbl]
        if len(position)>1:
            print('Warning: CustomTensorProcessing.py line 292, too many '
                  'positional labels. ')
        ipt[self.field_oi] = position[0]
        return ipt

class MTL_label(_TensorProcessing):
    def __init__(self, scheme, field_oi='y'):
        super(MTL_label, self).__init__()
        self.scheme = scheme
        self.field_oi = field_oi

    def __call__(self, ipt):
        ipt[self.field_oi] = eval(ipt[self.field_oi])

        res = {}
        for i, tsk in enumerate(self.scheme.keys()):
            try:
                res[tsk] = [self.scheme[tsk]({'y': ipt[self.field_oi][i]})['y']]
            except:
                res[tsk] = [0.5]
        ipt[self.field_oi] = res
        return ipt

class CORAL_Ordinal_label(_TensorProcessing):
    def __init__(self, ordinal_label, output_length=None, field_oi='y'):
        super(CORAL_Ordinal_label, self).__init__()
        self.ordinal_label = ordinal_label
        self.output_length = output_length
        self.field_oi = field_oi

    def __call__(self, ipt):
        ratios_of_ones = np.round(
            self.ordinal_label.index(ipt[self.field_oi]) * \
            self.output_length / (len(self.ordinal_label)-1)
        ).astype(int)

        ipt[self.field_oi] = ratios_of_ones * [1] + \
                             (self.output_length - ratios_of_ones) * [0]
        return ipt

class Softlabel_Ordinal_label(_TensorProcessing):
    def __init__(self, ordinal_label, output_length=None, field_oi='y',
                 metric_loss_function='Absolute', gaussian_std=5):
        super(Softlabel_Ordinal_label, self).__init__()
        self.ordinal_label = ordinal_label
        self.output_length = output_length
        self.metric_loss_function = metric_loss_function
        self.gaussian_std = gaussian_std
        self.field_oi = field_oi

    def __call__(self, ipt):
        if self.metric_loss_function=='Absolute':
            highest_spot = \
                self.ordinal_label.index(ipt[self.field_oi]) * \
                self.output_length / \
                (len(self.ordinal_label)-1)
            soft_func = [
                (self.output_length - 1) - np.abs((_ - highest_spot))
                for _ in range(self.output_length)
            ]
        elif self.metric_loss_function=='Gaussian':
            mean = \
                (self.ordinal_label.index(ipt[self.field_oi])+0.5) * \
                self.output_length / \
                (len(self.ordinal_label))
            def gaussian(x, mu, sig):
                return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
            soft_func = gaussian(np.array(
                [_ for _ in range(self.output_length)]
            ),
                mu=mean,
                sig = self.gaussian_std
            ).tolist()
        else:
            raise NotImplementedError(
                str(self.metric_loss_function) + ' is not a softlabel option. '
            )


        result = softmax([_ for _ in soft_func])

        ipt[self.field_oi] = result
        return ipt