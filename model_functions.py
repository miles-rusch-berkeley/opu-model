# See LICENSE.TT for license details.
from matplotlib._api import check_getitem
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

def area_utilization(M, N, K, ml, vl, kl):
    """
    average utilization of outer product array 
    over 3 dimensional volume of partial products
    edges and corners of volume do not fully utilize array
    """
    iNMK = 0
    util_a = 0
    tN = N
    while tN > 0:
        rN = min(vl, tN)
        tM = M
        while tM > 0:
            rM = min(ml, tM)
            tK = K
            while tK > 0:
                rK = min(kl, tK)
                util_a += rN*rM*rK
                tK = tK-rK
                iNMK += 1
            tM = tM-rM
        tN = tN-rN
    util_a /= iNMK
    util_a /= (vl*ml*kl)
    return util_a

def dataflow_model(databits, t_mem, M,N,K, l2_cache, kl, vlB, vl_ml, num_mregs, t_op_ind, widen, width_datapath):
    """
    From software parameters:
        databits: number of bits per vector element 
        widen: widening factor between input and output elements
    and microarchitecture parameters:
        t_mem: memory latency,  
        num_mregs: number of 2D matrix registers
        vlB: bytes per vector
        vl_ml: vl/ml, where ml is number of vectors per matrix register
        kl: number of outer product operations accumulated per instruction,
        t_op_ind: select functional unit latency
        width_datapath: half width reduces bw and increases latency both by a factor of four
        'l2_size': cache size in KB,
    Calculate model outputs:
        'mem_bw': average memory bandwidth for outer product BLAS schedule,
        'mrf_bw': matrix register file bandwidth
        't_uk': ukernel latency,
        'ops_cycle': macc operations per cycle,
        'mrf_capacity': matrix register file capacity
        'util': macc array utilization over ukernel
    """
    mlB = vlB / vl_ml
    ml = mlB/(databits/8) #num MMU rows equals number of elements ml
    vl = vlB/(databits/8)
    c_tile = ml * vlB*widen/kl**2

    # CACHE
    # double buffer B[kl * vlB] and C[ml * vlB]*nregs
    # a = mlB*(kc+kl) * num_mregs
    mc = min(M, num_mregs * ml)
    l2_cache_B = l2_cache*2**10
    kc = (l2_cache_B - 2*c_tile)/(mc * databits + vlB)
    kc = min(kc, K)
    l3_blas = N*kc*databits/2**23 #[MB]
    l3_nmk = N*K*databits/2**23 #[MB]

    #different opacc fu latencies
    t_op = [
        2*ml + kc*kl,
        ml + kc*kl,
        max(ml, kc*kl)
    ]
    t_crit = t_op[t_op_ind]/width_datapath**2
    t_uk = 2*t_mem + t_crit
    # number of parallel memory requests required to hide latency
    p_l2 = t_uk/t_crit
    max_mregs = math.ceil(p_l2)
    # effective ukernel latency given memory latency
    lsu_buffer = 1
    p_mrf = lsu_buffer * num_mregs/(databits/8)
    t_eff_opacc = max(t_uk/p_mrf, t_crit)
    
    # time utilization
    util_t = t_crit/t_eff_opacc
    # area utilization
    util_a = area_utilization(M, N, K, ml, vl, kl)
    # total utilization
    util = util_t*util_a
    #equivalent 8-Byte operations per cycle
    ops_cycle = util*ml*vlB/kl
    
    # Memory System
    # BLAS outer loops
    # 4th loop over K
    iKc = math.ceil(K/kc)
    b_blas = kc * N*databits/8
    # 3rd loop over M
    iMc = math.ceil(M/mc)
    a_blas = kc * mc*databits/8
    # 2nd loop over N
    iN = math.ceil(N/vl)
    c_tile = mc * vlB*widen/kl**2
    mem_lds = iKc*(b_blas + iMc*(a_blas + iN*c_tile))
    # 1st loop over mc
    iMi = math.ceil(mc/ml)
    # 0th loop over kc
    t_blas = iKc * iMc * iN * iMi * t_eff_opacc
    blas_mem_bw = mem_lds/t_blas

    #ukernel op intensity (opi) [ops/load]
    mrf_opi = num_mregs*vl*ml/kl**2/((num_mregs/2)*vl + 2*ml) if (num_mregs%2==0) else num_mregs*vl*ml/kl**2/(num_mregs*vl + ml)
    l2_opi = N * mc / (N + mc)

    # Matrix REGFILE
    md_c = ml * vlB*widen / kl**2
    mrf_bw = (c_tile + kc*(mlB + vlB))/t_eff_opacc
    max_mrf_capacity = max_mregs*c_tile
    mrf_capacity = num_mregs*md_c

    #from syn [um^2]
    top = 1764000   # total area
    rvv = 1373300   # vector backend area
    vrf = 82740     # vector register file area
    opu = 898150    # outer product unit area 
    mrf_byte = 256/4 * (8/32)   # 8-bit register area
    macc_byte = 250             # 8-bit MACC area
    l2_cache_128kB = 2*68590    # L2 cache area
    
    # scalar frontend area
    scalar_area = top - rvv - l2_cache_128kB
    #local vector cache area
    l2_area = l2_cache_128kB/128 * l2_cache
    # extrapolate to vector area
    vpu_area = (rvv - opu - vrf)*(vlB*width_datapath/32) + vrf*(vlB/64) + scalar_area
    # extrapolate to matrix area
    mrf_area = mrf_byte * mrf_capacity
    num_maccs = (vl*width_datapath) * (ml*width_datapath) / kl
    shift_area = mrf_byte * (databits/8) * widen * num_maccs
    opacc_area = macc_byte * (databits/8)**2 * num_maccs + shift_area
    opu_area =  l2_area + vpu_area + mrf_area + opacc_area
    
    #performance opu vs vpu
    t_vec_crit = 1 + kc*ml/width_datapath
    t_uk_vec = 2*t_mem + t_vec_crit
    t_eff_opacc_vec = max(t_uk_vec, ml*t_vec_crit)
    speedup_vec = t_eff_opacc_vec/t_eff_opacc

    perf_specs = {
        't_uk': t_uk,
        'util': util,
        'ops_cycle': ops_cycle,

        'max_mregs': max_mregs,
        'mrf_capacity': mrf_capacity/2**10,         # [kB]
        'max_mrf_capacity': max_mrf_capacity/2**10, # [kB]
        'l3_blas': l3_blas,
        'l3_nmk': l3_nmk,
        'blas_mem_bw': blas_mem_bw,
        'mrf_opi': mrf_opi,
        'l2_opi': l2_opi,
        'mrf_bw': mrf_bw,
        
        'mrf_area': mrf_area,
        'opacc_area': opacc_area,
        'opu_area': opu_area,

        'speedup_vec': speedup_vec,
        'vpu_area': vpu_area,
    }
    return perf_specs


# def generate_df(databits, M,N,K, mlB,vlB,kl, t_mem, flow_key):
def generate_df(databits, t_mem, M,N,K, l2_cache, kl, vlB, mlB, num_mregs, t_op, widen, width_datapath):
    # Create the df index space
    index_space = [databits, t_mem, M, N, K, l2_cache, kl, vlB, mlB, num_mregs, t_op,  widen, width_datapath]
    index_labels = ['databits', 't_mem', 'M','N','K', 'l2_cache', 'kl', 'vlB', 'vl_ml', 'num_mregs', 't_op',  'widen', 'width_datapath']
    # define df index over all possible combinations of input elements (cross product)
    df_index = pd.MultiIndex.from_product(index_space, names=index_labels)
    # Create columns of  DataFrame 
    df_columns = ['t_uk', 'util',
                  'ops_cycle', 'max_mregs', 'max_mrf_capacity',
                  'blas_mem_bw', 'mrf_bw', 
                  'l2_opi', 'mrf_opi',
                  'mrf_capacity', 'l3_blas', 'l3_nmk',
                  'speedup_vec', 'vpu_area',
                  'opacc_area', 'mrf_area', 'opu_area']
    df = pd.DataFrame(index=df_index, columns=df_columns,dtype=float)

    #compute performance specs
    for idx in tqdm(df_index, disable=True):
        perf_specs = dataflow_model(*idx)
        df.loc[idx, 't_uk'] = perf_specs['t_uk']
        df.loc[idx, 'util'] = perf_specs['util']
        df.loc[idx, 'ops_cycle'] = perf_specs['ops_cycle']
        
        df.loc[idx, 'max_mregs'] = perf_specs['max_mregs']
        df.loc[idx, 'max_mrf_capacity'] = perf_specs['max_mrf_capacity']
        df.loc[idx, 'l3_blas'] = perf_specs['l3_blas']
        df.loc[idx, 'l3_nmk'] = perf_specs['l3_nmk']
        df.loc[idx, 'blas_mem_bw'] = perf_specs['blas_mem_bw']
        df.loc[idx, 'l2_opi'] = perf_specs['l2_opi']
        df.loc[idx, 'mrf_opi'] = perf_specs['mrf_opi']

        df.loc[idx, 'mrf_bw'] = perf_specs['mrf_bw']
        df.loc[idx, 'mrf_capacity'] = perf_specs['mrf_capacity']
        df.loc[idx, 'mrf_area'] = perf_specs['mrf_area']
        df.loc[idx, 'opacc_area'] = perf_specs['opacc_area']
        df.loc[idx, 'opu_area'] = perf_specs['opu_area']
        df.loc[idx, 'speedup_vec'] = perf_specs['speedup_vec']
        df.loc[idx, 'vpu_area'] = perf_specs['vpu_area']
    return df

# Use `init_pm` to initialize model with desired input ranges. Defaults are scalars to allow for easy sweeping of one variable.
def init_pm(
    databits = np.array([8]),
    widen = np.array([4]),      
    t_mem = np.array([20]),     # [cycles]
    M = np.array([64]),         # [num elements]
    N = np.array([64]),         # [num elements]
    K = np.array([64]),         # [num elements]
    l2_cache = np.array([128]), # [KBytes]
    vlB = np.array([256])/8,    # [Bytes]
    vl_ml = np.array([1]),    # [Bytes]
    kl = np.array([1]),         # [num rows]
    num_mregs = np.array([4]),
    t_op = np.array([1]),     # [cycles]
    width_datapath = np.array([0.5]),     # [1 or 1/2]
    ):
    return generate_df(databits, t_mem, M,N,K, l2_cache, kl, vlB, vl_ml, num_mregs, t_op, widen, width_datapath)
