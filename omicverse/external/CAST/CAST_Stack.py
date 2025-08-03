import torch,copy,os,random
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from dataclasses import dataclass, field
from .visualize import add_scale_bar

#################### Registration ####################
# Parameters class

@dataclass
class reg_params:
    dataname : str
    ### affine
    theta_r1 : float = 0
    theta_r2 : float = 0
    d_list : list[float] = field(default_factory=list)
    translation_params : list[float] = None
    mirror_t : list[float] = None
    alpha_basis : list[float] = field(default_factory=list)
    iterations : int = 500
    dist_penalty1 : float = 0
    attention_params: list[float] = field(default_factory=list)

    ### BS
    mesh_trans_list : list[float] = field(default_factory=list)
    attention_region : list[float] = field(default_factory=list)
    attention_params_bs : list[float] = field(default_factory=list)
    mesh_weight : list[float] = field(default_factory=list)
    iterations_bs : list[float] = field(default_factory=list)
    alpha_basis_bs : list[float] = field(default_factory=list)
    meshsize : list[float] = field(default_factory=list)
    img_size_bs : list[float] = field(default_factory=list) # max_xy
    dist_penalty2 : list[float] = field(default_factory=list)
    PaddingRate_bs : float = 0

    ### common
    bleeding : float = 500
    diff_step : float = 5
    min_qr2 : float = 0
    mean_q : float = 0
    mean_r : float = 0
    gpu: int = 0
    device : str = field(init=False)
    ifrigid : bool = False
    
    def __post_init__(self):
        if self.gpu != -1 and torch.cuda.is_available():
            self.device = 'cuda:{}'.format(self.gpu)
        else:
            self.device = 'cpu'

def get_range(sp_coords):
    yrng = max(sp_coords, key=lambda x:x[1])[1] - min(sp_coords, key=lambda x:x[1])[1]
    xrng = max(sp_coords, key=lambda x:x[0])[0] - min(sp_coords, key=lambda x:x[0])[0]
    return xrng, yrng

def prelocate(coords_q,coords_r,cov_anchor_it,bleeding,output_path,d_list=[1,2,3],prefix = 'test',ifplot = True,index_list = None,translation_params = None,mirror_t = None):
    idx_q = np.ones(coords_q.shape[0],dtype=bool) if index_list is None else index_list[0]
    idx_r = np.ones(coords_r.shape[0],dtype=bool) if index_list is None else index_list[1]
    mirror_t = [1,-1] if mirror_t is None else mirror_t
    theta_t = []
    J_t = []
    if translation_params is None:
        translation_x = [0]
        translation_y = [0]
    else:
        xrng, yrng = get_range(coords_r.detach().cpu())
        dx_ratio_max, dy_ratio_max, xy_steps = translation_params
        dx_max = dx_ratio_max * xrng
        dy_max = dy_ratio_max * yrng
        translation_x = np.linspace(-dx_max, dx_max, num=int(xy_steps)) # dx
        translation_y = np.linspace(-dy_max, dy_max, num=int(xy_steps)) # dy
    for mirror in mirror_t:
        for dx in translation_x:
            for dy in translation_y:
                for d in d_list:
                    for phi in [0,90,180,270]:
                        a = d
                        d = d * mirror
                        theta = torch.Tensor([a,d,phi,dx,dy]).reshape(5,1).to(coords_q.device)
                        coords_query_it = affine_trans_t(theta,coords_q)
                        try:
                            J_t.append(J_cal(coords_query_it[idx_q],coords_r[idx_r],cov_anchor_it,bleeding).sum().item())
                        except:
                            continue
                        theta_t.append(theta)
    if ifplot:
        prelocate_loss_plot(J_t,output_path,prefix)
    return(theta_t[np.argmin(J_t)])

def Affine_GD(coords_query_it_raw,coords_ref_it,cov_anchor_it,output_path,bleeding=500, dist_penalty = 0,diff_step = 50,alpha_basis = np.reshape(np.array([0,0,1/5,2,2]),[5,1]),iterations = 50,prefix='test',attention_params = [None,3,1,0],scale_t = 1,coords_log = False,index_list = None, mid_visual = False,early_stop_thres = 1, ifrigid = False):
    idx_q = np.ones(coords_query_it_raw.shape[0],dtype=bool) if index_list is None else index_list[0]
    idx_r = np.ones(coords_ref_it.shape[0],dtype=bool) if index_list is None else index_list[1]
    dev = coords_query_it_raw.device
    theta = torch.Tensor([1,1,0,0,0]).reshape(5,1).to(dev) # initial theta, [a,d,phi,t1,t2]
    coords_query_it = coords_query_it_raw.clone()
    plot_mid(coords_query_it.cpu() * scale_t,coords_ref_it.cpu() * scale_t,output_path,prefix + '_init',scale_bar_t=None) if mid_visual else None
    similarity_score = [J_cal(coords_query_it[idx_q],coords_ref_it[idx_r],cov_anchor_it,bleeding,dist_penalty,attention_params).sum().cpu().item()]
    it_J = []
    it_theta = []
    coords_q_log = []
    delta_similarity_score = [np.inf] * 5
    t = trange(iterations, desc='', leave=True)
    for it in t:
        alpha = alpha_init(alpha_basis,it,dev)
        ## de_sscore
        dJ_dxy_mat = dJ_dt_cal(coords_query_it[idx_q],
                      coords_ref_it[idx_r],
                      diff_step,
                      dev,
                      cov_anchor_it,
                      bleeding,
                      dist_penalty,
                      attention_params)

        dJ_dtheta = dJ_dtheta_cal(coords_query_it[idx_q,0],
                                  coords_query_it[idx_q,1],
                                  dJ_dxy_mat,theta,dev,ifrigid = ifrigid)
        theta = theta_renew(theta,dJ_dtheta,alpha,ifrigid = ifrigid)

        coords_query_it = affine_trans_t(theta,coords_query_it_raw)
        it_J.append(dJ_dtheta)
        it_theta.append(theta)
        if coords_log:
            coords_q_log.append(coords_query_it.detach().cpu().numpy())

        sscore_t = J_cal(coords_query_it[idx_q],coords_ref_it[idx_r],cov_anchor_it,bleeding,dist_penalty,attention_params).sum().cpu().item()
        # print(f'Loss: {sscore_t}')
        t.set_description(f'Loss: {sscore_t:.3f}')
        t.refresh()
        similarity_score.append(sscore_t)
        if mid_visual:
            if (it % 20 == 0) | (it == 0):
                plot_mid(coords_query_it.cpu() * scale_t,coords_ref_it.cpu() * scale_t,output_path,prefix + str(int(it/10 + 0.5)),scale_bar_t=None)
        if early_stop_thres is not None and it > 200:
            delta_similarity_score.append(similarity_score[-2] - similarity_score[-1])
            if np.all(np.array(delta_similarity_score[-5:]) < early_stop_thres):
                print(f'Early stop at {it}th iteration.')
                break
    return([similarity_score,it_J,it_theta,coords_q_log])

def BSpline_GD(coords_q,coords_r,cov_anchor_it,iterations,output_path,bleeding, dist_penalty = 0, alpha_basis = 1000,diff_step = 50,mesh_size = 5,prefix = 'test',mesh_weight = None,attention_params = [None,3,1,0],scale_t = 1,coords_log = False, index_list = None, mid_visual = False,max_xy = None,renew_mesh_trans = True,restriction_t = 0.5):
    idx_q = np.ones(coords_q.shape[0],dtype=bool) if index_list is None else index_list[0]
    idx_r = np.ones(coords_r.shape[0],dtype=bool) if index_list is None else index_list[1]
    dev = coords_q.device
    plot_mid(coords_q.cpu() * scale_t,coords_r.cpu()* scale_t,output_path,prefix + '_FFD_initial_' + str(iterations),scale_bar_t=None) if mid_visual else None

    max_xy = coords_q.max(0)[0].cpu() if max_xy is None else max_xy
    mesh,mesh_weight,kls,dxy_ffd_all,delta = BSpline_GD_preparation(max_xy,mesh_size,dev,mesh_weight)
    coords_query_it = coords_q.clone()

    similarity_score = [J_cal(coords_query_it[idx_q],coords_r[idx_r],cov_anchor_it,bleeding,dist_penalty,attention_params).sum().cpu().item()]
    mesh_trans_list = []
    coords_q_log = []
    mesh_trans = mesh.clone()
    max_movement = (max_xy / (mesh_size - 1.) * restriction_t).to(mesh.device).unsqueeze(-1).unsqueeze(-1)
    t = trange(iterations, desc='', leave=True)
    for it in t:
        dJ_dxy_mat = dJ_dt_cal(coords_query_it[idx_q],
                coords_r[idx_r],
                diff_step,
                dev,
                cov_anchor_it,
                bleeding,
                dist_penalty,
                attention_params)
        if renew_mesh_trans or it == 0:
            uv_raw, ij_raw = BSpline_GD_uv_ij_calculate(coords_query_it,delta,dev)
            uv = uv_raw[:,idx_q] # 2 * N[idx]
            ij = ij_raw[:,idx_q] # 2 * N[idx]

        result_B_t = B_matrix(uv,kls) ## 16 * N[idx]
        dxy_ffd = get_dxy_ffd(ij,result_B_t,mesh,dJ_dxy_mat,mesh_weight,alpha_basis)
        
        if renew_mesh_trans:
            mesh_trans = mesh + dxy_ffd
        else:
            mesh_trans = mesh + torch.clamp(mesh_trans + dxy_ffd - mesh, min=-max_movement, max=max_movement)
        mesh_trans_list.append(mesh_trans)
        coords_query_it = BSpline_renew_coords(uv_raw,kls,ij_raw,mesh_trans)
        if coords_log:
            coords_q_log.append(coords_query_it.detach().cpu().numpy())
        sscore_t = J_cal(coords_query_it[idx_q],coords_r[idx_r],cov_anchor_it,bleeding,dist_penalty,attention_params).sum().cpu().item()
        # print(f'Loss: {sscore_t}')
        t.set_description(f'Loss: {sscore_t:.3f}')
        t.refresh()

        similarity_score.append(sscore_t)
        if mid_visual:
            if (it % 20 == 0) | (it == 0):
                plot_mid(coords_query_it.cpu() * scale_t,coords_r.cpu() * scale_t,output_path,prefix + '_FFD_it_' + str(it),scale_bar_t=None)
                mesh_plot(mesh.cpu(),coords_q_t=coords_query_it.cpu(),mesh_trans_t=mesh_trans.cpu())
                plt.savefig(f'{output_path}/{prefix}_mesh_plot_it_{it}.pdf')
                plt.clf()
    ### visualization
    plt.figure(figsize=[20,10])
    plt.subplot(1,2,1)
    plt.scatter(np.array(coords_q.cpu()[:,0].tolist()) * scale_t,
        np.array(coords_q.cpu()[:,1].tolist()) * scale_t,  s=2,edgecolors='none', alpha = 0.5,rasterized=True,
        c='blue',label = 'Before')
    plt.scatter(np.array(coords_query_it.cpu()[:,0].tolist()) * scale_t,
        np.array(coords_query_it.cpu()[:,1].tolist()) * scale_t, s=2,edgecolors='none', alpha = 0.7,rasterized=True,
        c='#ef233c',label = 'After')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=15)
    plt.axis('equal')
    plt.subplot(1,2,2)
    titles = 'loss = ' + format(similarity_score[-1],'.1f')
    plt.scatter(list(range(0,len(similarity_score))),similarity_score,s = 5)
    plt.title(titles,fontsize=20)
    plt.savefig(os.path.join(output_path,prefix + '_after_Bspline_' + str(iterations) + '.pdf'))
    return([coords_query_it,mesh_trans_list,dxy_ffd_all,similarity_score,coords_q_log])

def J_cal(coords_q,coords_r,cov_mat,bleeding = 10, dist_penalty = 0,attention_params = [None,3,1,0]):
    attention_region,double_penalty,penalty_inc_all,penalty_inc_both = attention_params
    bleeding_x = coords_q[:, 0].min() - bleeding, coords_q[:, 0].max() + bleeding
    bleeding_y = coords_q[:, 1].min() - bleeding, coords_q[:, 1].max() + bleeding

    sub_ind = ((coords_r[:, 0] > bleeding_x[0]) & (coords_r[:, 0] < bleeding_x[1]) &
               (coords_r[:, 1] > bleeding_y[0]) & (coords_r[:, 1] < bleeding_y[1]))
    
    cov_mat_t = cov_mat[:,sub_ind]
    dist = torch.cdist(coords_q,coords_r[sub_ind,:])
    min_dist_values, close_idx = torch.min(dist, dim=1)

    tmp1 = torch.stack((torch.arange(coords_q.shape[0], device=coords_q.device), close_idx)).T
    s_score_mat = cov_mat_t[tmp1[:, 0], tmp1[:, 1]]
    
    if(dist_penalty != 0):
        penalty_tres = torch.sqrt((coords_r[:,0].max() - coords_r[:,0].min()) * (coords_r[:,1].max() - coords_r[:,1].min()) / coords_r.shape[0])
        dist_d = min_dist_values / penalty_tres
        if(type(attention_region) is np.ndarray):
            attention_region = torch.tensor(attention_region, device=coords_q.device)
            dist_d[attention_region] = min_dist_values[attention_region] / (penalty_tres/double_penalty)
            dist_d[dist_d < 1] = 1
            dist_d[dist_d > 1] *= dist_penalty
            dist_d[attention_region] *= penalty_inc_all
            dist_d[(dist_d > 1) & attention_region] *= (penalty_inc_both/dist_penalty + 1)
        else:
            dist_d[dist_d < 1] = 1
            dist_d[dist_d > 1] *= dist_penalty
        return s_score_mat * dist_d
    return s_score_mat

def alpha_init(alpha_basis,it,dev):
    return 5/torch.pow(torch.Tensor([it/40 + 1]).to(dev),0.6) * alpha_basis

def dJ_dt_cal(coords_q,coords_r,diff_step,dev,cov_anchor_it,bleeding,dist_penalty,attention_params):
    dJ_dy = (J_cal(coords_q + torch.tensor([0,diff_step], device=dev),
                          coords_r,
                          cov_anchor_it,
                          bleeding,
                          dist_penalty,
                          attention_params) - 
               J_cal(coords_q + torch.tensor([0,-diff_step], device=dev),
                          coords_r,
                          cov_anchor_it,
                          bleeding,
                          dist_penalty,
                          attention_params)) / (2 * diff_step)
    dJ_dx = (J_cal(coords_q + torch.tensor([diff_step,0], device=dev),
                          coords_r,
                          cov_anchor_it,
                          bleeding,
                          dist_penalty,
                          attention_params) - 
               J_cal(coords_q + torch.tensor([-diff_step,0], device=dev),
                          coords_r,
                          cov_anchor_it,
                          bleeding,
                          dist_penalty,
                          attention_params)) / (2 * diff_step)
    dJ_dxy_mat = torch.vstack((dJ_dx,dJ_dy)) # [dJ_{i}/dx_{i},dJ_{i}/dy_{i}] (2 * N)
    return dJ_dxy_mat

def dJ_dtheta_cal(xi,yi,dJ_dxy_mat,theta,dev,ifrigid = False):
    '''
    #dxy_da:
    #{x * cos(rad_phi), x * sin(rad_phi)}
    #dxy_dd:
    #{-y * sin(rad_phi), y * cos(rad_phi)}
    #dxy_dphi:
    #{-d * y * cos(rad_phi) - a * x * sin(rad_phi), a * x * cos(rad_phi) - d * y * sin(rad_phi)}
    #dxy_dt1:
    #{1, 0}
    #dxy_dt2:
    #{0, 1}

    # when we set d = a (rigid):
    #dxy_da 
    #{x * cos(rad_phi) - y * sin(rad_phi), y * cos(rad_phi) + x * sin(rad_phi)}
    #dxy_dd - set as the same value as dxy_da
    #{x * cos(rad_phi) - y * sin(rad_phi), y * cos(rad_phi) + x * sin(rad_phi)}
    #dxy_dphi
    #{-a * y * cos(rad_phi) - a * x * sin(rad_phi), a * x * cos(rad_phi) - a * y * sin(rad_phi)}
    '''
    N = xi.shape[0]
    rad_phi = theta[2,0].deg2rad()
    cos_rad_phi = rad_phi.cos()
    sin_rad_phi = rad_phi.sin()
    ones = torch.ones(N, device=dev)
    zeros = torch.zeros(N, device=dev)
    if ifrigid:
        #### let d = a, only allow scaling, rotation and translation (Similarity transformation)
        #### If we want to use pure rigid transformation, just set `alpha_basis` as `[0,0,x,x,x]`, then the theta[0] will be always 1.
        dxy_dtheta = torch.stack([
            torch.stack([
                xi * cos_rad_phi - yi * sin_rad_phi, #dxy_da (rigid)
                xi * cos_rad_phi - yi * sin_rad_phi, #dxy_dd - won't use (rigid)
                -theta[0] * cos_rad_phi * yi - theta[0] * xi * sin_rad_phi, #dxy_dphi
                ones, #dxy_dt1
                zeros]), #dxy_dt2
            torch.stack([
                yi * cos_rad_phi + xi * sin_rad_phi, #dxy_da (rigid)
                yi * cos_rad_phi + xi * sin_rad_phi, #dxy_dd - won't use (rigid)
                theta[0] * xi * cos_rad_phi - theta[0] * yi * sin_rad_phi, #dxy_dphi
                zeros, #dxy_dt1
                ones])]) #dxy_dt2
    else:
        dxy_dtheta = torch.stack([
            torch.stack([
                xi * cos_rad_phi, #dxy_da
                -yi * sin_rad_phi, #dxy_dd
                -theta[1] * cos_rad_phi * yi - theta[0] * xi * sin_rad_phi, #dxy_dphi
                ones, #dxy_dt1
                zeros]), #dxy_dt2
            torch.stack([
                xi * sin_rad_phi, #dxy_da
                yi * cos_rad_phi, #dxy_dd
                theta[0] * xi * cos_rad_phi - theta[1] * yi * sin_rad_phi, #dxy_dphi
                zeros, #dxy_dt1
                ones])]) #dxy_dt2

    dJ_dtheta = torch.bmm(dxy_dtheta.permute(2, 1, 0), ### [N,5,2]
                          dJ_dxy_mat.transpose(0, 1).unsqueeze(-1) ### [N,2,1]
                          ).squeeze(2) # [dJ_{i}/dtheta_{k}] (N * 5)
    dJ_dtheta = dJ_dtheta.sum(0)

    return dJ_dtheta

def theta_renew(theta,dJ_dtheta,alpha,ifrigid = False):
    alpha_dJ = alpha * dJ_dtheta.reshape(5,1)
    alpha_dJ[0:3] = alpha_dJ[0:3] / 1000 # avoid dtheta_{abcd} change a lot of x and y
    if ifrigid & (theta[0] == -theta[1]):
        # only when the rigid transformation is allowed, we should check the value of d and a if they are mirrored.
        # if d and a are mirrored (setting in the prelocate `d = d * mirror``), we should set alpha_dJ[1] as the `-alpha_dJ[1]`.
        alpha_dJ[1] = -alpha_dJ[1] 
    theta_new = theta - alpha_dJ
    return theta_new

def affine_trans_t(theta,coords_t):
    rad_phi = theta[2,0].deg2rad()
    cos_rad_phi = rad_phi.cos()
    sin_rad_phi = rad_phi.sin()
    A = torch.Tensor([[theta[0,0] * cos_rad_phi, -theta[1,0] * sin_rad_phi],[theta[0,0] * sin_rad_phi, theta[1,0] * cos_rad_phi]]).to(theta.device)
    t_vec = theta[3:5,:]
    coords_t1 = torch.mm(A,coords_t.T) + t_vec
    coords_t1 = coords_t1.T
    return coords_t1

def torch_Bspline(uv, kl):
    return (
        torch.where(kl == 0, (1 - uv) ** 3 / 6,
        torch.where(kl == 1, uv ** 3 / 2 - uv ** 2 + 2 / 3,
        torch.where(kl == 2, (-3 * uv ** 3 + 3 * uv ** 2 + 3 * uv + 1) / 6,
        torch.where(kl == 3, uv ** 3 / 6, torch.zeros_like(uv)))))
    )

def BSpline_GD_preparation(max_xy,mesh_size,dev,mesh_weight):
    delta = max_xy / (mesh_size - 1.)
    mesh = np.ones((2, mesh_size + 3, mesh_size + 3)) ## 2 * (mesh_size + 3) * (mesh_size + 3)
    for i in range(mesh_size + 3):
        for j in range(mesh_size + 3):
            mesh[:, i, j] = [(i - 1) * delta[0], (j - 1) * delta[1]] ## 0 - -delta, 1 - 0, 2 - delta, ..., 6 - delta * 5, 7 - delta * 6 (last row)
    mesh = torch.tensor(mesh).to(dev)
    mesh_weight = torch.tensor(mesh_weight).to(dev) if type(mesh_weight) is np.ndarray else 1
    kls = torch.stack(torch.meshgrid(torch.arange(4), torch.arange(4))).flatten(1).to(dev) ## 2 * 16
    dxy_ffd_all = torch.zeros(mesh.shape, device=dev) ## 2 * (mesh_size + 3) * (mesh_size + 3)
    return mesh,mesh_weight,kls,dxy_ffd_all,delta

def BSpline_GD_uv_ij_calculate(coords_query_it,delta,dev):
    pos_reg = coords_query_it.T / delta.reshape(2,1).to(dev) # 2 * N
    pos_floor = pos_reg.floor().long() # 2 * N
    uv_raw = pos_reg - pos_floor # 2 * N
    ij_raw = pos_floor - 1 # 2 * N
    return uv_raw, ij_raw

def B_matrix(uv_t, kls_t):
    result_B_list = []
    for kl in kls_t.T:
        B = torch_Bspline(uv_t, kl.view(2, 1)) # 2 * N[idx]
        result_B_list.append(B.prod(0, keepdim=True)) # 1 * N[idx] ; .prod() - product of all elements in the tensor along a given dimension (0 - reduce along rows, 1 - reduce along columns)
    return torch.cat(result_B_list,0) # 16 * N[idx]

def get_dxy_ffd(ij,result_B_t,mesh,dJ_dxy_mat,mesh_weight,alpha_basis):
    dxy_ffd_t = torch.zeros(mesh.shape, device=result_B_t.device)
    ij_0 = ij[0] + 1
    ij_1 = ij[1] + 1
    for k in range(dxy_ffd_t.shape[1]):
        for l in range(dxy_ffd_t.shape[2]):
            mask = (ij_0 <= k) & (k <= ij_0 + 3) & (ij_1 <= l) & (l <= ij_1 + 3)
            if mask.any():  # check if there is any True in the mask
                idx_kl = mask.nonzero().flatten()
                ij_t = torch.tensor([k, l], device=ij.device) - (ij[:, idx_kl].T + 1)
                keys = ij_t[:, 0] * 4 + ij_t[:, 1]
                t33 = result_B_t[keys, idx_kl]
                dxy_ffd_t[:,k,l] -= torch.matmul(dJ_dxy_mat[:,idx_kl],t33.unsqueeze(1).float()).squeeze(1)
    dxy_ffd_t *= mesh_weight
    dxy_ffd_t = dxy_ffd_t * alpha_basis
    return dxy_ffd_t

def BSpline_renew_coords(uv_t,kls_t,ij_t,mesh_trans):
    result_tt = torch.zeros_like(uv_t, dtype=torch.float32)
    for kl in kls_t.T:
        B = torch_Bspline(uv_t, kl.view(2, 1))
        pivots = (ij_t + 1 + kl.view(2, 1)).clamp(0, mesh_trans.size(-1) - 1)
        mesh_t = mesh_trans[:, pivots[0], pivots[1]]
        result_tt += B.prod(0, keepdim=True) * mesh_t
    return result_tt.T

def reg_total_t(coords_q,coords_r,params_dist):
    dev = params_dist.device
    mean_q = coords_q.mean(0)
    mean_r = coords_r.mean(0)
    coords_q_t = torch.tensor(np.array(coords_q) - mean_q).float().to(dev) ## Initial location
    coords_q_r1 = affine_trans_t(params_dist.theta_r1,coords_q_t) ## Prelocation 1st Affine
    coords_q_r2 = affine_trans_t(params_dist.theta_r2,coords_q_r1) ## Affine transformation 2st Affine
    if params_dist.mesh_trans_list != [] and params_dist.mesh_trans_list != [[]]:
        coords_q_r3 = coords_q_r2.clone()
        for round_t in range(len(params_dist.mesh_trans_list)):
            coords_q_r3 = coords_q_r3.clone() - params_dist.min_qr2[round_t]
            coords_q_r3 = FFD_Bspline_apply_t(coords_q_r3.clone(),params_dist,round_t)
            coords_q_r3 = coords_q_r3.clone() + params_dist.min_qr2[round_t]
        coords_q_f = coords_q_r3.clone()
    else:
        coords_q_f = coords_q_r2
    coords_q_reconstruct = coords_q_f + torch.tensor(mean_r).to(dev)
    coords_q_reconstruct = coords_q_reconstruct.float()
    return coords_q_f,coords_q_reconstruct

def FFD_Bspline_apply_t(coords_q,params_dist,round_t = 0):
    mesh_trans_list = params_dist.mesh_trans_list[round_t]
    dev = coords_q.device
    img_size = params_dist.img_size_bs[round_t]
    mesh_size = mesh_trans_list[0].shape[2] - 3
    delta = img_size / (mesh_size - 1.)
    coords_query_it = copy.deepcopy(coords_q)

    for it in trange(len(mesh_trans_list), desc='', leave=True):
        mesh_trans = mesh_trans_list[it]
        pos_reg = coords_query_it.T / delta.reshape(2,1).to(dev)
        pos_floor = pos_reg.floor().long()
        uv = pos_reg - pos_floor
        ij = pos_floor - 1
        kls = torch.stack(torch.meshgrid(torch.arange(4), torch.arange(4))).flatten(1).to(dev)
        result_tt = torch.zeros_like(uv).float()
        for kl in kls.T:
            B = torch_Bspline(uv, kl.view(2, 1))
            pivots = (ij + 1 + kl.view(2, 1)).clamp(0, mesh_trans.size(-1) - 1)
            mesh_t = mesh_trans[:, pivots[0], pivots[1]]
            result_tt += B.prod(0, keepdim=True) * mesh_t
        coords_query_it = result_tt.T
    return coords_query_it

def rescale_coords(coords_raw,graph_list,rescale = False):
    rescale_factor = 1
    if rescale:
        coords_raw = coords_raw.copy()
        for sample_t in graph_list:
            rescale_factor_t = 22340 / np.abs(coords_raw[sample_t]).max()
            coords_raw[sample_t] = coords_raw[sample_t].copy() * rescale_factor_t
            if sample_t == graph_list[1]:
                rescale_factor = rescale_factor_t
    return coords_raw,rescale_factor

#################### Visualization ####################

def mesh_plot(mesh_t,coords_q_t,mesh_trans_t = None):
    mesh_no_last_row = mesh_t[:, :, :].numpy()
    plt.figure(figsize=[10,10])
    plt.plot(mesh_no_last_row[0], mesh_no_last_row[1], 'blue')
    plt.plot(mesh_no_last_row.T[..., 0], mesh_no_last_row.T[..., 1], 'blue')
    if(type(mesh_trans_t) is not type(None)):
        mesh_trans_no_last_row = mesh_trans_t[:, :, :].numpy()
        plt.plot(mesh_trans_no_last_row[0], mesh_trans_no_last_row[1], 'orange')
        plt.plot(mesh_trans_no_last_row.T[..., 0], mesh_trans_no_last_row.T[..., 1], 'orange')
    plt.scatter(coords_q_t.T[0,:],coords_q_t.T[1,:],c='blue',s = 0.5,alpha=0.5, rasterized=True)

def plot_mid(coords_q,coords_r,output_path='',filename = None,title_t = ['ref','query'],s_t = 8,scale_bar_t = None):
    plt.rcParams.update({'font.size' : 30,'axes.titlesize' : 30,'pdf.fonttype':42,'legend.markerscale' : 5})
    plt.figure(figsize=[10,12])
    plt.scatter(np.array(coords_r)[:,0].tolist(),
        np.array(coords_r)[:,1].tolist(),  s=s_t,edgecolors='none', alpha = 0.5,rasterized=True,
        c='#9295CA',label = title_t[0])
    plt.scatter(np.array(coords_q)[:,0].tolist(),
        np.array(coords_q)[:,1].tolist(), s=s_t,edgecolors='none', alpha = 0.5,rasterized=True,
        c='#E66665',label = title_t[1])
    plt.legend(fontsize=15)
    plt.axis('equal')
    if (type(scale_bar_t) != type(None)):
        add_scale_bar(scale_bar_t[0],scale_bar_t[1])
    if (filename != None):
        plt.savefig(os.path.join(output_path,filename + '.pdf'),dpi = 100)

def corr_heat(coords_q,coords_r,corr,output_path,title_t = ['Corr in ref','Anchor in query'],filename=None,scale_bar_t = None):
    plt.rcParams.update({'font.size' : 20,'axes.titlesize' : 20,'pdf.fonttype':42})
    random.seed(2)
    sampled_points = np.sort(random.sample(list(range(0,coords_q.shape[0])),20))
    plt.figure(figsize=((40,25)))
    for t in range(0,len(sampled_points)):

        plt_ind = t * 2
        ins_cell_idx = sampled_points[t]
        col_value = corr[ins_cell_idx,:]
        col_value_bg = [0] * coords_q.shape[0]
        col_value_bg[ins_cell_idx] = 1
        size_value_bg = [5] * coords_q.shape[0]
        size_value_bg[ins_cell_idx] = 30
        plt.subplot(5,8,plt_ind + 1)
        plt.scatter(np.array(coords_r[:,0]), np.array(coords_r[:,1]), s=5,edgecolors='none',
            c=col_value,cmap = 'vlag',vmin = -1,vmax= 1,rasterized=True)

        plt.title(title_t[0])
        plt.axis('equal')
        if (type(scale_bar_t) != type(None)):
            add_scale_bar(scale_bar_t[0],scale_bar_t[1])
        plt.subplot(5,8,plt_ind + 2)
        plt.scatter(np.array(coords_q[:,0]), np.array(coords_q[:,1]), s=size_value_bg,edgecolors='none',
            c=col_value_bg,cmap = 'vlag',vmin = -1,vmax= 1,rasterized=True)
        plt.scatter(np.array(coords_q[ins_cell_idx,0]), np.array(coords_q[ins_cell_idx,1]), s=size_value_bg[ins_cell_idx],edgecolors='none',
            c=col_value_bg[ins_cell_idx],cmap = 'vlag',vmin = -1,vmax= 1,rasterized=True)
        plt.title(title_t[1])
        plt.axis('equal')
        if (type(scale_bar_t) != type(None)):
            add_scale_bar(scale_bar_t[0],scale_bar_t[1])
    plt.tight_layout()
    plt.colorbar()
    if (filename != None):
        plt.savefig(os.path.join(output_path,filename + '.pdf'),dpi=100,transparent=True)

def prelocate_loss_plot(J_t,output_path,prefix = 'test'):
    plt.rcParams.update({'font.size' : 15})
    plt.figure(figsize=[5,5])
    plt.scatter(x=list(range(0,len(J_t))),y=J_t)
    plt.savefig(f'{output_path}/{prefix}_prelocate_loss.pdf')

def register_result(coords_q,coords_r,cov_anchor_t,bleeding,embed_stack,output_path,k=8,prefix='test',scale_t = 1,index_list = None):
    idx_q = np.ones(coords_q.shape[0],dtype=bool) if index_list is None else index_list[0]
    idx_r = np.ones(coords_r.shape[0],dtype=bool) if index_list is None else index_list[1]
    coords_q = coords_q * scale_t
    coords_r = coords_r * scale_t
    kmeans = KMeans(n_clusters=k,random_state=0).fit(embed_stack)
    cell_label = kmeans.labels_
    cluster_pl = sns.color_palette('tab20',len(np.unique(cell_label)))
    ### panel 1 ###
    plot_mid(coords_q[idx_q],coords_r[idx_r],output_path,f'{prefix}_Results_1', scale_bar_t = None)
    ### panel 2 ###
    plt.figure(figsize=[10,12])
    plt.rcParams.update({'font.size' : 10,'axes.titlesize' : 20,'pdf.fonttype':42})
    col=coords_q[idx_q,0]
    row=coords_q[idx_q,1]
    cell_type_t = cell_label[0:coords_q[idx_q].shape[0]]
    for i in set(cell_type_t):
        plt.scatter(np.array(col)[cell_type_t == i],
        np.array(row)[cell_type_t == i], s=12,edgecolors='none',alpha = 0.5,rasterized=True,
        c=np.array(cluster_pl)[cell_type_t[cell_type_t == i]],label = str(i))
    col=coords_r[idx_r,0]
    row=coords_r[idx_r,1]
    cell_type_t = cell_label[coords_q[idx_q].shape[0]:]
    for i in set(cell_type_t):
        plt.scatter(np.array(col)[cell_type_t == i],
        np.array(row)[cell_type_t == i], s=12,edgecolors='none',alpha = 0.5,rasterized=True,
        c=np.array(cluster_pl)[cell_type_t[cell_type_t == i]],label = str(i))
    plt.axis('equal')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('K means (k = ' + str(k) + ')',fontsize=30)
    add_scale_bar(200,'200 µm')
    plt.savefig(f'{output_path}/{prefix}_Results_2.pdf',dpi = 300)
    ### panel 3 ###
    plt.figure(figsize=[20,12])
    plt.subplot(1,2,1)
    t_score = J_cal(torch.from_numpy(coords_q[idx_q]),torch.from_numpy(coords_r[idx_r]),cov_anchor_t,bleeding)
    plt.scatter(coords_q[idx_q,0],coords_q[idx_q,1],c=1 - t_score,cmap = 'vlag',vmin = -1,vmax = 1,s = 15,edgecolors='none',alpha=0.5,rasterized=True)
    add_scale_bar(200,'200 µm')
    plt.subplot(1,2,2)
    plt.scatter(coords_q[0,0],coords_q[0,1],c=[0],cmap = 'vlag',vmin = -1,vmax = 1,s = 15,alpha=0.5)
    plt.axis('off')
    plt.colorbar()
    plt.savefig(f'{output_path}/{prefix}_Results_3.pdf',dpi = 300)

def affine_reg_params(it_theta,similarity_score,iterations,output_path,prefix='test'):
    plt.rcParams.update({'font.size' : 15,'axes.titlesize' : 15,'pdf.fonttype':42})
    similarity_score_t = copy.deepcopy(similarity_score)
    titles = ['a','d','φ','t1','t2','loss = ' + format(similarity_score[-1],'.1f')]
    plt.figure(figsize=[15,8])
    for i in range(0,6):
        plt.subplot(2,4,i+1)
        if i == 5:
            plt.scatter(list(range(0,len(similarity_score_t))),similarity_score_t,s = 5)
        else:
            # plt.scatter(x = range(0,iterations),y=np.array(it_theta)[:,i,0],s = 5)
            plt.scatter(x = range(1,len(similarity_score_t)),y=np.array(it_theta)[:,i,0],s = 5)
        plt.title(titles[i],fontsize=20)
    plt.savefig(os.path.join(output_path,prefix + '_params_Affine_GD_' + str(iterations) + 'its.pdf'))

def CAST_STACK_rough(coords_raw_list, ifsquare=True, if_max_xy=True, percentile = None):
    '''
    coords_raw_list: list of numpy arrays, each array is the coordinates of a layer
    ifsquare: if True, the coordinates will be scaled to a square
    if_max_xy: if True, the coordinates will be scaled to the max value of the `max_range_x` and `max_range_y`, respectively (if ifsquare is False), or the max value of [max_range_x,max_range_y] (if ifsquare is True)
    percentile: if not None, the min and max will be calculated based on the percentile of the coordinates for each slice.
    '''
    # Convert list of arrays to a single numpy array for easier processing
    all_coords = np.concatenate(coords_raw_list)
    # Finding the global min and max for both x and y
    if percentile is None:
        min_x, min_y = np.min(all_coords, axis=0)
        max_x, max_y = np.max(all_coords, axis=0)
    else:
        min_x_list, min_y_list, max_x_list, max_y_list = [], [], [], []
        for coords_t in coords_raw_list:
            min_x_list.append(np.percentile(coords_t[:,0],percentile))
            min_y_list.append(np.percentile(coords_t[:,1],percentile))
            max_x_list.append(np.percentile(coords_t[:,0],100-percentile))
            max_y_list.append(np.percentile(coords_t[:,1],100-percentile))
        min_x, min_y = np.min(min_x_list), np.min(min_y_list)
        max_x, max_y = np.max(max_x_list), np.max(max_y_list)
    max_xy = np.array([max_x - min_x, max_y - min_y])
    scaled_coords_list = []
    for coords_t in coords_raw_list:
        coords_t2 = (coords_t - coords_t.min(axis=0)) / np.ptp(coords_t, axis=0)
        if if_max_xy:
            max_xy_scale = max_xy
        else:
            max_xy_scale = max_xy / np.max(max_xy)
        scaled_coords = coords_t2 * np.max(max_xy_scale) if ifsquare else coords_t2 * max_xy_scale
        scaled_coords_list.append(scaled_coords)
    return scaled_coords_list

#################### Calculation ####################
def coords_minus_mean(coord_t):
    return np.array(coord_t) - np.mean(np.array(coord_t),axis = 0)

def coords_minus_min(coord_t):
    return np.array(coord_t) - np.min(np.array(coord_t),axis = 0)

def max_minus_value(corr):
    return np.max(corr) - corr

def coords_minus_min_t(coord_t):
    return coord_t - coord_t.min(0)[0]

def max_minus_value_t(corr):
    return corr.max() - corr

def corr_dist(query_np, ref_np, nan_as = 'min'):
    from sklearn.metrics import pairwise_distances_chunked
    def chunked_callback(dist_matrix,start):
        return 1 - dist_matrix
    chunks = pairwise_distances_chunked(query_np, ref_np, metric='correlation', n_jobs=-1, working_memory=1024, reduce_func=chunked_callback)
    corr_q_r = np.vstack(list(chunks))
    if nan_as == 'min':
        corr_q_r[np.isnan(corr_q_r)] = np.nanmin(corr_q_r)
    return corr_q_r

def region_detect(embed_dict_t,coords0,k = 20):
    plot_row = int(np.floor((k+1)/4) + 1)
    kmeans = KMeans(n_clusters=k,random_state=0).fit(embed_dict_t)
    cell_label = kmeans.labels_
    cluster_pl = sns.color_palette('tab20',len(np.unique(cell_label)))
    plt.figure(figsize=((20,5 * plot_row)))
    plt.subplot(plot_row,4,1)
    cell_label_idx = 0
    col=coords0[:,0].tolist()
    row=coords0[:,1].tolist()
    cell_type_t = cell_label[cell_label_idx:(cell_label_idx + coords0.shape[0])]
    cell_label_idx += coords0.shape[0]
    for i in set(cell_type_t):
        plt.scatter(np.array(col)[cell_type_t == i],
        np.array(row)[cell_type_t == i], s=5,edgecolors='none',
        c=np.array(cluster_pl)[cell_type_t[cell_type_t == i]],label = str(i))
    plt.title(' (KMeans, k = ' + str(k) + ')',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.axis('equal')
    for j,i in enumerate(set(cell_type_t)):
        plt.subplot(plot_row,4,j+2)
        plt.scatter(np.array(col),np.array(row),s=3,c = '#DDDDDD')
        plt.scatter(np.array(col)[cell_type_t == i],
        np.array(row)[cell_type_t == i], s=5,edgecolors='none',
        c=np.array(cluster_pl)[cell_type_t[cell_type_t == i]],label = str(i))
        plt.title(str(i),fontsize=20)
        plt.axis('equal')
    return cell_label
