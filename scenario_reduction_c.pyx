from __future__ import division

import cython
import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel
from libc.math cimport sqrt
from libc.stdlib cimport malloc, free


@cython.boundscheck(False)
@cython.wraparound(False)
def cost_matrix_c_par(double[::1,:] scenarios, int num_threads, str cost_func='2norm', int r=1, double[:] scen0=None,
                      int verbose=0):
    # calculate cost matrix using parallel processing
    cdef double [:,:] costmatrix

    if verbose > 0:
        print('Starting Generation of cost Matrix, with dist_func = {}'.format(cost_func))

    if cost_func == '2norm':
        costmatrix = calc_cost_matrix_2norm(scenarios, num_threads)
    elif cost_func == '1norm':
        costmatrix = calc_cost_matrix_1norm(scenarios, num_threads)
    elif cost_func == 'abs_max':
        costmatrix = calc_cost_matrix_absmax(scenarios, num_threads)
    elif cost_func == 'General':
        costmatrix = calc_cost_matrix_general(scenarios, r, scen0, num_threads)
    else:
        print('Distance function c not implemented, please provide an implemented cost function c')
        return None

    if verbose > 0:
        print('Generation of cost Matrix done')

    return costmatrix


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:,:] calc_cost_matrix_2norm(double[::1,:] scenarios, int num_threads=1):
    cdef int n_sc = scenarios.shape[1]
    cdef int nt = scenarios.shape[0]

    cdef double[:,:] c = np.empty((n_sc ,n_sc), dtype=np.float64)

    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t k

    cdef double norm

    for i in prange(n_sc, nogil=True, num_threads=num_threads):
        for j in range(i +1 ,n_sc):
            norm = 0
            for k in range(nt):
                norm += (scenarios[k,i] - scenarios[k,j])**2
            norm = sqrt(norm)
            c[i,j] = norm
            c[j,i] = norm
    return c


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:,:] calc_cost_matrix_general(double[::1,:] scenarios, int r, double[:] scen0=None, int num_threads=1):
    cdef int n_sc = scenarios.shape[1]
    cdef int nt = scenarios.shape[0]

    cdef double[:,:] c = np.empty((n_sc ,n_sc), dtype=np.float64)
    if scen0 is None:
        scen0 = np.zeros(n_sc, dtype= np.float64)

    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t k

    cdef double norm_diff
    cdef double norm1
    cdef double norm2
    cdef double c_i

    for i in prange(n_sc, nogil=True, num_threads=num_threads):
        for j in range(i +1 ,n_sc):
            norm_diff = 0
            norm1 = 0
            norm2 = 0
            for k in range(nt):
                norm_diff += (scenarios[k,i] - scenarios[k,j])**2
                norm1 += (scenarios[k,i] - scen0[k])**2
                norm2 += (scenarios[k,i] - scen0[k])**2
            norm_diff = sqrt(norm_diff)
            norm1 = sqrt(norm1)**(r-1)
            norm2 = sqrt(norm2)**(r-1)
            if (norm1 > norm2) and (norm1 > 1):
                c_i = norm_diff*norm1
            elif (norm2 > norm1) and (norm2 > 1):
                c_i = norm_diff*norm2
            else:
                c_i = norm_diff
            c[i,j] = c_i
            c[j,i] = c_i
    return c


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:,:] calc_cost_matrix_1norm(double[::1,:] scenarios, int num_threads=1):
    cdef int n_sc = scenarios.shape[1]
    cdef int nt = scenarios.shape[0]

    cdef double[:,:] c = np.empty((n_sc ,n_sc), dtype=np.float64)

    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t k

    cdef double norm

    for i in prange(n_sc, nogil=True, num_threads=num_threads):
        for j in range(i +1 ,n_sc):
            norm = 0
            for k in range(nt):
                if scenarios[k,i] > scenarios[k,j]:
                    norm += scenarios[k,i] - scenarios[k,j]
                else:
                    norm += scenarios[k,j] - scenarios[k,i]
            norm = norm
            c[i,j] = norm
            c[j,i] = norm
    return c


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:,:] calc_cost_matrix_absmax(double[::1,:] scenarios, int num_threads=1):
    cdef int n_sc = scenarios.shape[1]
    cdef int nt = scenarios.shape[0]

    cdef double[:,:] c = np.empty((n_sc ,n_sc), dtype=np.float64)

    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t k

    cdef double max1
    cdef double max2
    cdef double c_i

    for i in prange(n_sc, nogil=True, num_threads=num_threads):
        for j in range(i +1 ,n_sc):
            max1 = 0
            max2 = 0
            c_i = 0
            for k in range(nt):
                if scenarios[k,i] > max1:
                    max1 = scenarios[k,i]
                if scenarios[k,j] > max2:
                    max2 = scenarios[k,j]
            c_i = max1-max2
            if c_i < 0:
                c_i = -c_i
            c[i,j] = c_i
            c[j,i] = c_i
    return c


@cython.boundscheck(False)
@cython.wraparound(False)
def fast_forward_sel_c_par(double[::1,:] scenarios, int n_sc_red, double[:,:] c, double[:] probs=None,
                           int num_threads=1, int verbose=0):
    # from 'Heitsch, Holger, and Werner Römisch. "Scenario reduction algorithms in stochastic programming."
    # Computational optimization and applications 24.2-3 (2003): 187-206.'
    # p.192, algorithm 2.2

    cdef int n_sc = scenarios.shape[1]
    if probs is None:
        probs = 1 / n_sc * np.ones(n_sc)
    cdef int n_sc_del = n_sc - n_sc_red

    cdef int[::1] idx_keep = np.empty(n_sc_red,dtype=np.int32)
    cdef int[::1] idx_del = np.arange(n_sc,dtype=np.int32)

    cdef Py_ssize_t k = 0
    cdef Py_ssize_t u
    cdef Py_ssize_t iu

    cdef double min_d_temp = 100000
    cdef double min_u_temp = 100000
    cdef int u_min_u_temp = 0
    cdef int iu_min_u_temp = 0
    cdef double sum_temp = 0
    cdef int size_idx_del = n_sc

    cdef double[:] sum_temps = np.empty(n_sc,dtype=np.float64)
    cdef double[:] dstart = 100000*np.ones(n_sc,dtype=np.float64)

    # step 1
    for iu in prange(size_idx_del, nogil=True, num_threads=num_threads):
        sum_temps[iu] = inner_loop_forw(size_idx_del, iu, idx_del, idx_keep, c, k, probs, dstart)

    for iu in range(size_idx_del):
        if sum_temps[iu] < min_u_temp:
            min_u_temp = sum_temps[iu]
            u_min_u_temp = idx_del[iu]
            iu_min_u_temp = iu

    idx_keep[k] = u_min_u_temp
    idx_del[iu_min_u_temp:n_sc-1] = idx_del[iu_min_u_temp+1:]
    idx_del[n_sc-1] = 0
    size_idx_del -= 1
    k += 1


    # step i
    while k < n_sc_red:
        if verbose >0:
            print('Selected {} scenarios'.format(k))
        min_u_temp = 100000
        for iu in prange(size_idx_del, nogil=True, num_threads=num_threads):
            sum_temps[iu] = inner_loop_forw(size_idx_del, iu, idx_del, idx_keep, c, k, probs, c[:,u_min_u_temp])

        for iu in range(size_idx_del):
            if sum_temps[iu] < min_u_temp:
                min_u_temp = sum_temps[iu]
                u_min_u_temp = idx_del[iu]
                iu_min_u_temp = iu

        idx_keep[k] = u_min_u_temp
        idx_del[iu_min_u_temp:n_sc-1] = idx_del[iu_min_u_temp+1:]
        idx_del[n_sc-1] = 0
        size_idx_del -= 1
        k += 1

    return np.array(idx_del[:n_sc_del]), np.array(idx_keep)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double inner_loop_forw(int size_idx_del, int iu, int[:] idx_del, int[:] idx_keep, double[:,:] c,
                            int k, double[:] probs, double[:] c_u_prev) nogil:
    cdef double sumtemp = 0

    cdef Py_ssize_t j
    cdef Py_ssize_t ij
    cdef Py_ssize_t i
    cdef Py_ssize_t ii
    cdef Py_ssize_t u

    u = idx_del[iu]
    for ij in range(size_idx_del-1):
        if ij < iu:
            j = idx_del[ij]
        else:
            j = idx_del[ij+1]

        if c_u_prev[j] < c[j,u]:
            c[j,u] = c_u_prev[j]

        sumtemp += probs[j] * c[j,u]

    return sumtemp



@cython.boundscheck(False)
@cython.wraparound(False)
def simult_backward_red_c_par(double[::1,:] scenarios, int n_sc_red, double[:,:] c, double[:] probs=None,
                              int num_threads=1, int verbose=0):
    # from 'Heitsch, Holger, and Werner Römisch. "Scenario reduction algorithms in stochastic programming."
    # Computational optimization and applications 24.2-3 (2003): 187-206.'
    # p.192, algorithm 2.2

    cdef int n_sc = scenarios.shape[1]
    if probs is None:
        probs = 1 / n_sc * np.ones(n_sc)
    cdef int n_sc_del = n_sc - n_sc_red

    cdef int[::1] idx_del = np.empty(n_sc_del,dtype=np.int)
    cdef int[::1] idx_keep = np.arange(n_sc,dtype=np.int)

    cdef Py_ssize_t k = 0
    cdef Py_ssize_t l
    cdef Py_ssize_t il

    cdef double min_l_temp = 100000
    cdef int l_min_l_temp = 0
    cdef int il_min_l_temp = 0
    cdef int size_idx_keep = n_sc
    cdef double sum_temp = 0

    cdef double[:] sum_temps = np.empty(n_sc,dtype=np.float64)

    # step i
    while k < n_sc_del:
        if verbose > 0:
            print('Reduced {} scenarios'.format(k))
        for il in prange(size_idx_keep, nogil=True, num_threads=num_threads):
            l = idx_keep[il]
            sum_temps[il] = inner_loop_backward(size_idx_keep, idx_keep, k, idx_del, c, probs, l, il)

        min_l_temp = 100000
        for il in range(size_idx_keep):
            if sum_temps[il] < min_l_temp:
                min_l_temp = sum_temps[il]
                l_min_l_temp = idx_keep[il]
                il_min_l_temp = il

        idx_del[k] = l_min_l_temp

        idx_keep[il_min_l_temp:n_sc-1] = idx_keep[il_min_l_temp+1:]
        idx_keep[n_sc-1] = 0
        size_idx_keep += -1
        k += 1

    return idx_del, idx_keep[:n_sc_red]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double inner_loop_backward(int size_idx_keep, int[:] idx_keep, int k, int[:] idx_del, double[:,:] d,
                     double[:] probs, int l, int il) nogil:

    cdef double min_d_temp
    cdef int* idx_keep_no_l = <int*> malloc(sizeof(int) * size_idx_keep)
    cdef double sum_temp = 0

    cdef Py_ssize_t j
    cdef Py_ssize_t ij
    cdef Py_ssize_t i
    cdef Py_ssize_t ii

    try: # put in try block to avoid memory leak in case of errors
        for ii in range(size_idx_keep-1):
            if ii < il:
                idx_keep_no_l[ii] = idx_keep[ii]
            else:
                idx_keep_no_l[ii] = idx_keep[ii+1]

        # cases j = idx_del
        for ij in range(k):
            j = idx_del[ij]
            min_d_temp = 100000

            for ii in range(size_idx_keep-1):
                i = idx_keep_no_l[ii]
                if d[i,j] < min_d_temp:
                    min_d_temp = d[i,j]
            sum_temp += probs[j] * min_d_temp

        # case j = l
        min_d_temp = 100000
        for ii in range(size_idx_keep-1):
            i = idx_keep_no_l[ii]
            if d[i,l] < min_d_temp:
                min_d_temp = d[i,l]
        sum_temp += probs[l] * min_d_temp
    finally:
        free(idx_keep_no_l)

    return sum_temp




