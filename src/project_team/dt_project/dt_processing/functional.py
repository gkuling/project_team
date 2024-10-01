from project_team.project_config import is_Primitive

from copy import deepcopy
import torch

def reduce_by_sum_sitk(list_of_data, reduction_by='X_location'):
    all_sessions = list(set([v[reduction_by] for v in list_of_data]))
    return_res = []
    for ses in all_sessions:
        res = [v for v in list_of_data if v[reduction_by]==ses]
        if len(res)==1:
            return_res.append(res[0])
            continue
        result = deepcopy(res[0])
        for next in res[1:]:
            for k in result.keys():
                if is_Primitive(result[k]) and is_Primitive(next[k]) and \
                        is_Primitive(result[k]) == is_Primitive(next[k]):
                    result[k] = result[k]
                else:
                    if all([type(v)==sitk.Image for v in result[k]]) and all([
                        type(v)==sitk.Image for v in next[k]]):
                        result[k] = [sitk.Add(*v) for v
                                     in zip(result[k], next[k])]
                    else:
                        result[k] = [v for v in zip(result[k], next[k])]
        return_res.append(result)
    return return_res


def make_all_tensors_same_size(btch):
    """
    Combines batch files into a collated tensor for traininng. Can have a max
    sequence length that you wish to adhere to, or if None will make the max
    length the longest sequence in the batch
    :param btch: batch file recievevd from the data laoder
    :param max_len: len of sequence you wish for the batch to adhere to
    :return: batch with collated data
    """
    return_btch = {key:None for key in btch[0].keys()}
    for key in btch[0].keys():
        assert all([type(btch[0][key])==type(x[key]) for x in btch]), \
            "something has gone wrong and all the items for this key are not " \
            "the same type "
        if type(btch[0][key])==torch.Tensor:
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
