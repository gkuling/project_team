from project_team.project_config import is_Primitive
import SimpleITK as sitk
from copy import deepcopy

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