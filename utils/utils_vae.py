import torch
from pylab import *

# The KL 
def EQ_diag_standard_normal_logpdf(p_inv_Sigmaq_diag, p_muq):
    assert(p_inv_Sigmaq_diag.shape[0] == p_muq.shape[0])
    assert(p_inv_Sigmaq_diag.shape[1] == p_muq.shape[1])
    MB, K = p_inv_Sigmaq_diag.shape
    return -0.5 * ( (p_muq * p_muq).sum(1).reshape(MB,1,1) +  (1/p_inv_Sigmaq_diag).sum(1).reshape(MB,1,1) ) - K/2 * log(2*pi)

# -E_{P(omega)} [log P(omega)]
def diagonal_normal_entropy(p_inv_Sigmaq_diag, p_muq):
    assert(p_muq.shape[2]==1)
    MB, K = p_inv_Sigmaq_diag.shape
    return K/2 + K/2 * log(2*pi) - 0.5 * torch.log(p_inv_Sigmaq_diag).sum(1).reshape(MB,1,1)


def unintegrated_lower_bound_diagonal_rao_blackwell_v(sess_array, sess_mask, p_Psi, p_rho, p_omega, p_inv_Sigmaq_diag, p_muq):
    assert(p_muq.shape[2]==1)
    MB = sess_array.shape[0]
    T = sess_mask.sum(1).reshape(MB,1,1)
    return ((torch.matmul(p_Psi[sess_array,:],p_muq) + p_rho[sess_array]) * sess_mask[:,:,None]).sum((1,2)).reshape(MB,1,1) - T * torch.logsumexp((torch.matmul(p_Psi,p_omega) + p_rho),1).reshape(MB,1,1)  + EQ_diag_standard_normal_logpdf(p_inv_Sigmaq_diag, p_muq) + diagonal_normal_entropy(p_inv_Sigmaq_diag, p_muq)


def torch_JJ(zeta):
    return 1./(2.*zeta)*(1./(1+torch.exp(-zeta)) - 0.5)


def block_torch_update(p_C, p_a, p_xi, p_Psi, p_rho, device):
    MB = p_C.shape[0]
    T = p_C.sum(1).reshape(MB,1,1)
    P,K = p_Psi.shape

    # update p_inv_Sigmaq
    p_inv_Sigmaq = torch.eye(K, device=device) + 2 * T * torch.matmul(p_Psi.t(),(torch_JJ(p_xi)[:,:,None] * p_Psi))

    # p_muq
    p_Psi_v = torch.matmul(p_Psi.t(),p_C)
    p_muq = torch.matmul(torch.inverse(p_inv_Sigmaq), p_Psi_v-T*((0.5+2*(p_rho-p_a).t()*torch_JJ(p_xi))[:,:,None] * p_Psi).sum(1).reshape(MB,K,1) )

    p_a = ((0.5*(P/2-1)+(torch.matmul(torch_JJ(p_xi)[:,:,None] * p_Psi, p_muq)+torch_JJ(p_xi)[:,:,None] * p_rho).sum(1))/torch_JJ(p_xi).sum(1).reshape(MB,1)).reshape(MB)

    # p_xi
    AA=(torch.matmul(p_Psi,p_muq) + (p_rho - p_a).t().reshape(MB,P,1))**2
    BB=(torch.matmul(p_Psi, torch.inverse(p_inv_Sigmaq))*p_Psi).sum(2)
    p_xi = torch.sqrt(AA.reshape(MB,P)+BB)

    return p_inv_Sigmaq, p_muq, p_a, p_xi