from torch.utils.cpp_extension import load
import torch
import unignnconv


def UniGNNConvdeg(dl, in_feat, degE, degV):
    return unignnconv.unignnconvdeg(dl.group_key, dl.group_row, dl.group_start, dl.group_end, dl.H_T_csrptr, dl.H_T_colind, in_feat, degE, degV)

def UniGNNConv(dl, in_feat):
    return unignnconv.unignnconv(dl.group_key, dl.group_row, dl.group_start, dl.group_end, dl.H_T_csrptr, dl.H_T_colind, in_feat)
