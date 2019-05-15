from ellipy.gen.common.common import throw_error, abs_rel_compare, is_numeric
from typing import Callable, Union
import numpy as np
from numpy import linalg


def is_mat_not_deg(q_mat: np.ndarray, abs_tol: float = 0.0) -> bool:
    if abs_tol < 0.0:
        throw_error('wrongInput:abs_tolNegative', 'abs_tol is expected to be non-negative')
    if abs_tol == 0.0:
        return True
    else:
        min_sing = np.min(np.linalg.svd(q_mat, compute_uv=False))
        if min_sing < abs_tol:
            return False
        else:
            return True


def is_mat_pos_def(q_mat: np.ndarray, abs_tol: float = 0., is_flag_sem_def_on: bool = False) -> bool:
    if abs_tol < 0.:
        throw_error('wrongInput:abs_tolNegative', 'abs_tol is expected to be nonnegative')
    if not is_mat_symm(q_mat, abs_tol):
        throw_error('wrongInput:nonSymmMat', 'input matrix must be symmetric')
    eig_vec = np.linalg.eigvalsh(q_mat)
    min_eig = min(eig_vec)
    is_pos_def = True
    if is_flag_sem_def_on:
        if min_eig < -abs_tol:
            is_pos_def = False
    else:
        if min_eig <= abs_tol:
            is_pos_def = False
    return is_pos_def


def is_mat_symm(q_mat: np.ndarray, abs_tol: float = 0.) -> bool:
    if q_mat.ndim != 2:
        throw_error('wrongInput:nonSquareMat', 'q_mat should be a matrix')
    n_rows = q_mat.shape[0]
    n_cols = q_mat.shape[1]
    if n_rows != n_cols:
        throw_error('wrongInput:nonSquareMat', 'q_mat should be a square matrix')

    abs_func: Callable[[np.ndarray], np.ndarray] = lambda x: np.abs(x)
    is_symm, *_ = abs_rel_compare(q_mat, q_mat.T, abs_tol, None, abs_func)
    return is_symm


def mat_orth(src_mat: np.ndarray) -> np.ndarray:
    o_mat, src_r_mat = np.linalg.qr(src_mat, 'complete')
    is_neg_diag = np.diagonal(src_r_mat) < 0.
    is_neg_diag_vec = np.zeros(o_mat.shape[1], dtype=bool)
    is_neg_diag_vec[:is_neg_diag.size] = is_neg_diag
    o_mat[:, is_neg_diag_vec] = -o_mat[:, is_neg_diag_vec]
    return o_mat


def math_orth_col(src_mat: np.ndarray) -> np.ndarray:
    o_mat = mat_orth(src_mat)
    return o_mat[:, :src_mat.shape[1]]


def ml_orth_transl(src_mat: np.ndarray, dst_arr: np.ndarray) -> np.ndarray:
    n_elems = 1 if dst_arr.ndim <= 2 else dst_arr.shape[2]
    n_vecs = 1 if dst_arr.ndim <= 1 else dst_arr.shape[1]
    n_dims = dst_arr.shape[0]
    src_mat = src_mat.reshape((n_dims, n_vecs))
    dst_arr = dst_arr.reshape((n_dims, n_vecs, n_elems))
    o_arr = np.zeros((n_dims, n_dims, n_elems, n_vecs), dtype=np.float64)
    for i_vec in range(n_vecs):
        src_vec = np.expand_dims(src_mat[:, i_vec], axis=1)
        for i_elem in range(n_elems):
            dst_vec = np.expand_dims(dst_arr[:, i_vec, i_elem], axis=1)
            o_arr[:, :, i_elem, i_vec] = orth_transl(src_vec, dst_vec)
    return o_arr


def orth_transl(src_vec: np.ndarray, dst_vec: np.ndarray) -> np.ndarray:
    __ABS_TOL = 1e-7
    n_dims = dst_vec.shape[0]
    src_vec = try_treat_as_real(src_vec)
    dst_vec = try_treat_as_real(dst_vec)
    dst_squared_norm = np.sum(dst_vec * dst_vec)
    src_squared_norm = np.sum(src_vec * src_vec)

    if dst_squared_norm == 0.0:
        throw_error('wrongInput:dst_zero', 'destination vectors are expected to be non-zero')
    if src_squared_norm == 0.0:
        throw_error('wrongInput:src_zero', 'source vectors are expected to be non-zero')

    dst_vec = dst_vec / np.sqrt(dst_squared_norm)
    src_vec = src_vec / np.sqrt(src_squared_norm)

    scal_prod = np.sum(src_vec * dst_vec)
    s_val = np.sqrt(np.maximum(1.0 - scal_prod * scal_prod, 0.0))
    q_mat = np.zeros((n_dims, 2), dtype=np.float64)
    q_mat[:, 0] = np.squeeze(dst_vec)
    if np.abs(s_val) > __ABS_TOL:
        q_mat[:, 1] = np.squeeze((src_vec - scal_prod * dst_vec) / s_val)
    else:
        q_mat[:, 1] = 0.0

    s_mat = np.array([[scal_prod - 1.0, s_val], [-s_val, scal_prod - 1.0]], dtype=np.float64)
    o_mat = np.identity(n_dims, dtype=np.float64) + q_mat @ s_mat @ q_mat.T
    return o_mat


def orth_transl_haus(src_vec: np.ndarray, dst_vec: np.ndarray) -> np.ndarray:

    __MAX_TOL = 1e-14
    src_norm_vec = src_vec / np.sqrt(np.sum(src_vec ** 2))
    dst_norm_vec = dst_vec / np.sqrt(np.sum(dst_vec ** 2))
    w_vec = src_norm_vec - dst_norm_vec
    if w_vec.ndim == 1:
        w_vec = np.expand_dims(w_vec, 1)
    w_norm = np.sum(w_vec ** 2)
    if w_norm < __MAX_TOL:
        r_mult = 0
    else:
        r_mult = 2 / w_norm

    o_mat = np.eye(np.size(src_vec)) - r_mult * (w_vec @ w_vec.T)
    return o_mat


def orth_transl_max_dir(src_vec: np.ndarray, dst_vec: np.ndarray,
                        src_max_vec: np.ndarray, dst_max_vec: np.ndarray) -> np.ndarray:
    n_dims = src_vec.size
    if n_dims > 1:
        e_vec = np.append(np.ones((1,), dtype=np.float64), np.zeros((n_dims - 1,), dtype=np.float64))
        o_src_mat = orth_transl(e_vec, src_vec)
        o_dst_mat = orth_transl(e_vec, dst_vec)

        v_0_mat = o_src_mat[:, 1:]
        u_0_mat = o_dst_mat[:, 1:]
        a_vec = o_src_mat[:, 0]
        b_vec = o_dst_mat[:, 0]
        a1_vec = v_0_mat.T @ src_max_vec
        b1_vec = u_0_mat.T @ dst_max_vec
        o1_src_mat, r1_src_mat = np.linalg.qr(a1_vec.reshape(-1, 1), 'complete')
        o1_dst_mat, r1_dst_mat = np.linalg.qr(b1_vec.reshape(-1, 1), 'complete')
        if np.logical_xor(r1_src_mat[0, 0] > 0, r1_dst_mat[0, 0] > 0):
            o1_dst_mat[:, 0] = -o1_dst_mat[:, 0]
        o_mat = u_0_mat @ o1_dst_mat @ o1_src_mat.T @ v_0_mat.T + (b_vec.reshape(-1, 1)) * a_vec
    else:
        o_mat = np.sign(src_vec) * np.sign(dst_vec)
    return o_mat


def orth_transl_max_tr(src_vec: np.ndarray, dst_vec: np.ndarray, max_mat: np.ndarray) -> np.ndarray:

    n_dims = src_vec.shape[0]
    if n_dims == 1:
        return np.ones((1, 1))
    else:
        e_vec = np.append(np.ones((1, ), dtype=np.float64), np.zeros((n_dims - 1, ), dtype=np.float64))
        o_src_mat = orth_transl(e_vec, src_vec)
        o_dst_mat = orth_transl(e_vec, dst_vec)

        v_0_mat = o_src_mat.T[1:, :]
        u_0_mat = o_dst_mat[:, 1:]

        src_n_vec = np.expand_dims(o_src_mat[:, 0], 1)
        dst_n_vec = np.expand_dims(o_dst_mat[:, 0], 1)

        k_mat = v_0_mat @ max_mat @ u_0_mat
        m_mat, _, n_mat = np.linalg.svd(k_mat, full_matrices=True)
        sline_mat = (m_mat @ n_mat).T
        o_mat = u_0_mat @ sline_mat @ v_0_mat + dst_n_vec @ src_n_vec.T
        return o_mat


def orth_transl_qr(src_vec: np.ndarray, dst_vec: np.ndarray) -> np.ndarray:

    src_vec = try_treat_as_real(src_vec)

    dst_vec = try_treat_as_real(dst_vec)
    dst_squared_norm = np.sum(dst_vec * dst_vec)

    src_squared_norm = np.sum(src_vec * src_vec)

    if dst_squared_norm == 0.0:
        throw_error('wrongInput:dst_zero', 'destination vectors are expected to be non-zero')
    if src_squared_norm == 0.0:
        throw_error('wrongInput:src_zero', 'source vectors are expected to be non-zero')

    if src_vec.ndim == 1:
        src_vec = np.expand_dims(src_vec, 1)
    if dst_vec.ndim == 1:
        dst_vec = np.expand_dims(dst_vec, 1)

    n_dims = src_vec.shape[0]
    if n_dims == 1:
        o_mat = np.sign(src_vec @ dst_vec)
        return o_mat
    else:
        q_mat, r_mat = np.linalg.qr(np.hstack((dst_vec, src_vec)))
        cos_val = dst_vec.T @ src_vec / np.sqrt(dst_squared_norm * src_squared_norm)
        sin_val = -np.sqrt(1 - cos_val * cos_val)
        if r_mat[0, 0] * r_mat[1, 1] < 0.:
            sin_val = -sin_val

        qs_mat = np.zeros((n_dims, 2), dtype=np.float64)
        qs_mat[:, 0] = q_mat[:, 0] * (cos_val - 1) + q_mat[:, 1] * sin_val
        qs_mat[:, 1] = -q_mat[:, 0] * sin_val + q_mat[:, 1] * (cos_val - 1)
        o_mat = np.identity(n_dims, dtype=np.float64) + qs_mat @ q_mat.T
        return o_mat


def reg_pos_def_mat(inp_mat: np.ndarray, reg_tol: float) -> np.ndarray:
    if not(np.isscalar(reg_tol) and is_numeric(reg_tol) and np.real(reg_tol) > 0.0):
        throw_error('wrongInput:reg_tol', 'reg_tol must be a positive numeric scalar')
    reg_tol = try_treat_as_real(reg_tol)
    if not(is_mat_symm(inp_mat)):
        throw_error('wrongInput:inp_mat', 'matrix must be symmetric')
    d_mat, v_mat = np.linalg.eig(inp_mat)
    m_mat = np.diag(np.maximum(0.0, reg_tol-d_mat))
    m_mat = v_mat @ m_mat @ v_mat.T
    regular_mat = inp_mat + m_mat
    regular_mat = 0.5 * (regular_mat + regular_mat.T)
    return regular_mat


def sqrtm_pos(q_mat: np.ndarray, abs_tol: float = 0.) -> np.ndarray:
    if abs_tol < 0.:
        throw_error('wrongInput:abs_tolNegative', 'abs_tol is expected to be not-negative')
    if not is_mat_symm(q_mat, abs_tol):
        throw_error('wrongInput:nonSymmMat', 'input matrix must be symmetric')
    d_vec, v_mat = np.linalg.eigh(q_mat)
    if np.any(d_vec < -abs_tol):
        throw_error('wrongInput:notPosSemDef', 'input matrix is expected to be positive semi-definite')
    d_vec[d_vec < 0.] = 0.
    d_mat = np.diag(np.sqrt(d_vec))
    return v_mat @ d_mat @ v_mat.T


def try_treat_as_real(inp_mat:  Union[bool, int, float, complex, np.ndarray], tol_val: float = np.finfo(float).eps) \
        -> np.ndarray:
    if not(np.isscalar(tol_val) and is_numeric(tol_val) and tol_val > 0.0):
        throw_error('wrongInput:tol_val', 'tol_val must be a positive numeric scalar')
    if np.all(np.isreal(inp_mat)):
        return inp_mat
    else:
        img_inp_mat = inp_mat.imag
        if np.isscalar(img_inp_mat):
            norm_value = np.abs(img_inp_mat)
        else:
            norm_value = linalg.norm(img_inp_mat, np.inf)
        if norm_value < tol_val:
            return np.real(inp_mat.real)
        else:
            throw_error('wrongInput:inp_mat',
                        'Norm of imaginary part of source object = {}. It can not be more then tol_val = {}.'
                        .format(norm_value, tol_val))


def reg_mat(inp_mat: np.ndarray, reg_tol: float) -> np.ndarray:
    if not(np.isscalar(reg_tol) and is_numeric(reg_tol) and reg_tol > 0.):
            throw_error('wrongInput:reg_tol', 'reg_tol must be a positive numeric scalar')
    reg_tol = try_treat_as_real(reg_tol)
    u_mat, s_vec, v_mat = np.linalg.svd(inp_mat)
    s_mat = np.diag(np.maximum(s_vec, reg_tol))
    res_mat = u_mat @ s_mat @ v_mat
    return res_mat
