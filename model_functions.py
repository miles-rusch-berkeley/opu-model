# See LICENSE.TT for license details.
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

def dataflow_model(databits, t_mem, M,N,K, l2_cache, kl, vlB, vl_ml, num_mregs, t_op_ind, widen, width_mmu):
    """
    From software parameters
        databits: number of bits per vector element 
        widen: widening factor between input and output elements
        t_mem: memory latency,  
    and microarchitecture parameters
        num_regs: number of 2D matrix registers
        vlB, mlB: bytes per vector, vectors per matrix register
        kl: number of outer product operations accumulated per instruction,
        t_op_ind: select functional unit latency
        width_mmu: half width reduces bw and increases latency both by a factor of four
        'l2_size': cache size in KB,
    Calculate 
        'mem_bw': average memory bandwidth for outer product BLAS schedule,
        'mrf_bw': matrix register file bandwidth
        't_uk': ukernel latency,
        'ops_cycle': macc operations per cycle,
        'mrf_capacity': matrix register file capacity
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
    t_crit = t_op[t_op_ind]/width_mmu**2
    t_uk = 2*t_mem + t_crit
    # number of parallel memory requests required to hide latency
    p_l2 = t_uk/t_crit
    max_mregs = math.ceil(p_l2)
    # effective ukernel latency given memory latency
    lsu_buffer = 4
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
    c_blas = mc * vlB*widen/kl**2
    mem_lds = iKc*(b_blas + iMc*(a_blas + iN*c_blas))
    blas_mem_opi = N*M*K/mem_lds
    # 1st loop over mc
    iMi = math.ceil(mc/ml)
    # 0th loop over kc
    t_blas = iKc * iMc * iN * iMi * t_eff_opacc
    blas_mem_bw = mem_lds/t_blas
    
    # NMK schedule
    # 3rd loop over M
    a_tile = K * mc*databits/8
    # 2nd loop over N
    iN = math.ceil(N/vl)
    b_tile = K * vlB
    c_tile = mc * vlB*widen/kl**2
    # 1st loop over M
    mem_lds = iMc*(a_tile + iN*iMi*b_tile)
    nmk_mem_opi = N*M*K/mem_lds
    # 0th loop over K
    t_nmk = iMc * iN * iMi * t_eff_opacc 
    nmk_mem_bw = mem_lds/t_nmk

    # Matrix REGFILE
    ms_a = kl * mlB
    ms_b = kl * vlB
    md_c = ml * vlB*widen/kl**2
    mrf_bw = (c_tile + kc*(mlB + vlB))/t_eff_opacc
    max_mrf_capacity = max_mregs*c_tile
    max_vrf_capacity = max_mregs*(ms_a + ms_b)
    mrf_capacity = num_mregs*md_c
    # vrf_capacity = num_mregs*(ms_a + ms_b)

    #parameters from hardware synthesis
    mrf_4x32b_area = 256 #um^2
    macc8b_area =  251 #um^2
    readout_4x4x32b_area = 60 #um^2
    vrf_vl256_area = (46116 + 46737 + 2*22598)/3 #um^2
    #extrapolate to model parameters
    mrf_area = mrf_capacity * (mrf_4x32b_area/4/4)
    vrf_area = vrf_vl256_area/(256/8) * vlB
    macc_area = (macc8b_area)*(databits/8)**2
    readout_area = readout_4x4x32b_area/(4*4)
    num_cells = (vl/width_mmu) * (ml/width_mmu) / kl
    opu_area =  vrf_area + mrf_area + num_cells*(macc_area + readout_area)
    #compare to vector unit
    vec_area = vrf_area + (macc_area)*vl/width_mmu
    t_vec_crit = 1 + kc*ml/width_mmu
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
        'nmk_mem_bw': nmk_mem_bw,
        'nmk_mem_opi': nmk_mem_opi,
        'blas_mem_bw': blas_mem_bw,
        'blas_mem_opi': blas_mem_opi,
        'mrf_bw': mrf_bw,
        
        'mrf_area': mrf_area,
        'macc_area': macc_area,
        'opu_area': opu_area,

        'speedup_vec': speedup_vec,
        'vec_area': vec_area,
    }
    return perf_specs


# def generate_df(databits, M,N,K, mlB,vlB,kl, t_mem, flow_key):
def generate_df(databits, t_mem, M,N,K, l2_cache, kl, vlB, mlB, num_mregs, t_op, widen, width_mmu):
    # Create the df index space
    index_space = [databits, t_mem, M, N, K, l2_cache, kl, vlB, mlB, num_mregs, t_op,  widen, width_mmu]
    index_labels = ['databits', 't_mem', 'M','N','K', 'l2_cache', 'kl', 'vlB', 'vl_ml', 'num_mregs', 't_op',  'widen', 'width_mmu']
    # define df index over all possible combinations of input elements (cross product)
    df_index = pd.MultiIndex.from_product(index_space, names=index_labels)
    # Create columns of  DataFrame 
    df_columns = ['t_uk', 'util',
                  'ops_cycle', 'max_mregs', 'max_mrf_capacity',
                  'blas_mem_bw', 'nmk_mem_bw', 'mrf_bw', 
                  'blas_mem_opi', 'nmk_mem_opi',
                  'mrf_capacity', 'l3_blas', 'l3_nmk',
                  'speedup_vec', 'vec_area',
                  'macc_area', 'mrf_area', 'opu_area']
    df = pd.DataFrame(index=df_index, columns=df_columns,dtype=float)

    #compute performance specs
    # idxs = pd.MultiIndex.from_product(index_space, names=index_labels)
    for idx in tqdm(df_index, disable=True):
        perf_specs = dataflow_model(*idx)
        df.loc[idx, 't_uk'] = perf_specs['t_uk']
        # df.loc[idx, 'util_a'] = perf_specs['util_a']
        # df.loc[idx, 'util_t'] = perf_specs['util_t']
        df.loc[idx, 'util'] = perf_specs['util']
        df.loc[idx, 'ops_cycle'] = perf_specs['ops_cycle']
        
        df.loc[idx, 'max_mregs'] = perf_specs['max_mregs']
        df.loc[idx, 'max_mrf_capacity'] = perf_specs['max_mrf_capacity']
        df.loc[idx, 'l3_blas'] = perf_specs['l3_blas']
        df.loc[idx, 'l3_nmk'] = perf_specs['l3_nmk']
        df.loc[idx, 'nmk_mem_bw'] = perf_specs['nmk_mem_bw']
        df.loc[idx, 'nmk_mem_opi'] = perf_specs['nmk_mem_opi']
        df.loc[idx, 'blas_mem_bw'] = perf_specs['blas_mem_bw']
        df.loc[idx, 'blas_mem_opi'] = perf_specs['blas_mem_opi']

        df.loc[idx, 'mrf_bw'] = perf_specs['mrf_bw']
        df.loc[idx, 'mrf_capacity'] = perf_specs['mrf_capacity']
        df.loc[idx, 'mrf_area'] = perf_specs['mrf_area']
        df.loc[idx, 'macc_area'] = perf_specs['macc_area']
        df.loc[idx, 'opu_area'] = perf_specs['opu_area']
        df.loc[idx, 'speedup_vec'] = perf_specs['speedup_vec']
        df.loc[idx, 'vec_area'] = perf_specs['vec_area']
    return df

# Use `init_pm` to initialize model with desired input ranges. Defaults are scalars to allow for easy sweeping of one variable.
def init_pm(
    databits = np.array([8]),
    t_mem = np.array([20]),     # [cycles]
    M = np.array([64]),         # [num elements]
    N = np.array([64]),         # [num elements]
    K = np.array([64]),         # [num elements]
    l2_cache = np.array([128]), # [KBytes]
    kl = np.array([1]),         # [num rows]
    vlB = np.array([256])/8,    # [Bytes]
    vl_ml = np.array([1]),    # [Bytes]
    num_mregs = np.array([2]),
    t_op = np.array([1]),     # [cycles]
    widen = np.array([4]),      
    width_mmu = np.array([0.5]),     # [1 or 1/2]
    ):
    return generate_df(databits, t_mem, M,N,K, l2_cache, kl, vlB, vl_ml, num_mregs, t_op, widen, width_mmu)
