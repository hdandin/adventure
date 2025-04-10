#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tools for sparse matrix, not present or ineficient for my usage in scipy

Created on Wed Feb 28 09:49:32 2024
@author: nchevaug
"""
import numpy as np
import scipy as sp

def regular_block_diag_to_csr(block_diag):
    ''' from an np.array of shape (nblock, sblock, sblock) representing a block diagonal
    matrix A of shape (nblock*sblock) with block size sblock
    constuct the csr_array representing A.
    '''
    fun_name = regular_block_diag_to_csr.__name__
    if len(block_diag.shape) != 3:
        raise ValueError('Error in '+fun_name+' input block_diag must be 3d.')
    if block_diag.shape[1] != block_diag.shape[2]:
        raise ValueError(
            "Error in "+fun_name+
            "input bD must be 3d, with shape[1] == shape[2] (array of square matrices)")
    nblock = block_diag.shape[0]  #number of diagonal block
    sblock = block_diag.shape[1]  #size of the block
    col = np.repeat(np.arange(sblock*nblock).reshape((-1, sblock)), sblock, axis=0).flatten()
    lin = np.arange(sblock*nblock).repeat(sblock)
    return sp.sparse.csr_array((block_diag.flatten(), (col, lin)), (sblock*nblock, sblock*nblock))
