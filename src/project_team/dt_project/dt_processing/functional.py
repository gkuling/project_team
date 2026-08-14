import torch


def make_all_tensors_same_size(btch):
    """
    Combines batch files into a collated tensor for training. Pads every 1D
    tensor in the batch out to the longest sequence in the batch.
    :param btch: batch file received from the data loader
    :return: batch with collated data
    """
    return_btch = {key:None for key in btch[0].keys()}
    for key in btch[0].keys():
        if not all(isinstance(x[key], type(btch[0][key])) for x in btch):
            raise TypeError(
                "All items in the batch for key '" + str(key) + "' must "
                "share one type to be collated.")
        if isinstance(btch[0][key], torch.Tensor):
            if torch.stack([torch.tensor(x[key].shape) for x in btch]).shape[
                1]==0:
                return_btch[key] = torch.stack([x[key] for x in btch])
            elif torch.stack([torch.tensor(x[key].shape) for x in btch]).shape[
                1]==1:
                max_len = torch.max(
                    torch.stack([torch.tensor(x[key].shape) for x in btch]),
                    dim=0
                )[0]
                return_btch[key] = torch.stack(
                    [torch.nn.functional.pad(x[key],
                                             pad=(0, max_len - x[key].numel()),
                                             mode='constant',
                                             value=0)
                     for x in btch]
                )
            else:
                raise Exception('same_size is not programmed to deal with '
                                'vectors dim>1. Needs an edit')
        else:
            return_btch[key] = [x[key] for x in btch]
    return return_btch
