from simso.core.Criticality import Criticality

def get_C_i_LO_DVFS(C_i_LO, f_base, f_i_LO):
    return C_i_LO * f_base / f_i_LO

def get_C_i_HI_DVFS(C_i_LO, C_i_HI, f_base, f_i_LO, f_i_HI):
    return (C_i_LO * f_base / f_i_LO) + ((C_i_HI - C_i_LO) * f_base) / f_i_HI

def get_U(C_DVFS_set, taskset):
    U = 0
    for task in taskset:
            U += C_DVFS_set[task] / task.period
    return U

def get_X_LB(U_HI_LO, U_LO_LO):
    return U_HI_LO / (1 - U_LO_LO)

def get_X_UB(U_HI_HI, U_LO_LO):
    return min(1, (1 - U_HI_HI) / U_LO_LO)

def get_X_LB_under_freq(taskset, f_base, f_LO_LO, f_HI_LO):
    taskset_LO = [task for task in taskset if task.criticality == Criticality.LO]
    taskset_HI = [task for task in taskset if task.criticality == Criticality.HI]
    C_i_LO_all_DVFS = {}
    for task in taskset:
        if task.criticality == Criticality.LO:
            C_i_LO_all_DVFS[task] = get_C_i_LO_DVFS(task.wcet, f_base, f_LO_LO)
        else:
            C_i_LO_all_DVFS[task] = get_C_i_LO_DVFS(task.wcet, f_base, f_HI_LO)
    U_HI_LO = get_U(C_i_LO_all_DVFS, taskset_HI)
    U_LO_LO = get_U(C_i_LO_all_DVFS, taskset_LO)
    return get_X_LB(U_HI_LO, U_LO_LO)

def get_X_UB_under_freq(taskset, f_base, f_LO_LO, f_HI_LO, f_HI_HI):
    taskset_LO = [task for task in taskset if task.criticality == Criticality.LO]
    taskset_HI = [task for task in taskset if task.criticality == Criticality.HI]
    C_i_LO_all_DVFS = {}
    C_i_HI_all_DVFS = {}
    for task in taskset_LO:
        C_i_LO_all_DVFS[task] = get_C_i_LO_DVFS(task.wcet, f_base, f_LO_LO)
    for task in taskset_HI:
        C_i_HI_all_DVFS[task] = get_C_i_HI_DVFS(task.wcet, task.wcet_high, f_base, f_HI_LO, f_HI_HI)
    U_HI_HI = get_U(C_i_HI_all_DVFS, taskset_HI)
    U_LO_LO = get_U(C_i_LO_all_DVFS, taskset_LO)
    return get_X_UB(U_HI_HI, U_LO_LO)

def get_K(taskset, f_base):
    taskset_HI = [task for task in taskset if task.criticality == Criticality.HI]
    K = 0
    for task in taskset_HI:
        K += task.wcet * f_base / task.period
    return K

def get_L(taskset, f_base):
    taskset_LO = [task for task in taskset if task.criticality == Criticality.LO]
    L = 0
    for task in taskset_LO:
        L += task.wcet * f_base / task.period
    return L

def get_M(taskset, f_base, f_max):
    taskset_HI = [task for task in taskset if task.criticality == Criticality.HI]
    M = 1
    for task in taskset_HI:
        M -= ((task.wcet_high - task.wcet) * f_base) / (task.period * f_max)
    return M


def static_optimal(taskset, f_base, f_min, f_max, alpha):

    x_LB = get_X_LB_under_freq(taskset, f_base, f_max, f_max)
    x_UB = get_X_UB_under_freq(taskset, f_base, f_max, f_max, f_max)

    f_HI_HI_opt = 1
    f_HI_LO_opt = 1
    f_LO_LO_opt = 1
    x_opt = 1

    if 0 < x_LB and x_LB < x_UB and x_UB <= 1:
        f_HI_HI_opt = f_max
        x_LB_ = get_X_LB_under_freq(taskset, f_base, f_min, f_min)
        x_UB_ = get_X_UB_under_freq(taskset, f_base, f_min, f_min, f_max)
        # print(x_LB_, x_UB_)

        if 0 < x_LB_ and x_LB_ <= x_UB_ and x_UB_ <= 1:
            f_LO_LO_opt = f_min
            f_HI_LO_opt = f_min
            x_opt = x_LB_
        else:
            K = get_K(taskset, f_base)
            L = get_L(taskset, f_base)
            M = get_M(taskset, f_base, f_max)
            # print('KLM', K, L, M)
            f_LO_LO_down = max(f_min, L / (1 - (K / (M * f_max))))
            
            if f_min > K / M:
                f_LO_LO_up = min(f_max, L / (1 - (K / (M * f_min))))
            else:
                f_LO_LO_up = f_max
            
            x_opt = M
            # print(f_LO_LO_down, f_LO_LO_up, K * pow(M, (-(alpha - 1) / alpha)) + L)
            f_LO_LO_opt = min(max(K * pow(M, (-(alpha - 1) / alpha)) + L, f_LO_LO_down), f_LO_LO_up)
            f_HI_LO_opt = K / (M * (1 - L / f_LO_LO_opt))
        
        return [
            f_LO_LO_opt,
            f_HI_LO_opt,
            f_HI_HI_opt,
            x_opt
        ]

    else:
        return None
    