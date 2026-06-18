"""GMORL MEC 环境 + SLA 扩展 (自包含, 不依赖 tianshou/gym).

源自 Generalizable-Pareto-Optimal-Offloading (GMORL) 的 Env.py, 改动:
  1. 删掉 tianshou/gym/tensorboard 等无用 import, env 本身只需 numpy。
  2. 给每个任务加 assign_step / tid, 在 exe 完成时收集 finished_task。
  3. 计算端到端时延 (off_time+exe_time)、SLA 违约、ω-无关的 SLA 惩罚通道 r_C。
  4. step() 返回 info 里带 3 通道向量 r_vec=[r_T, r_E, r_C] 与 SLA 统计。
原始 GMORL Env.py 保留在 _ref_gmorl/Env.py 作对照, 永不修改。
"""
import numpy as np
from copy import deepcopy
import json
import math

ACTION_TO_CLOUD = 0

RAYLEIGH_VAR = 1
RAYLEIGH_PATH_LOSS_A = 35
RAYLEIGH_PATH_LOSS_B = 133.6
RAYLEIGH_ANTENNA_GAIN = 0
RAYLEIGH_SHADOW_FADING = 8
RAYLEIGH_NOISE_dBm = -174

ZERO_RES = 1e-6
MAX_EDGE_NUM = 10
        
class MEC_Env():
    def __init__(self, conf_file='config_sla.json', conf_name='MEC_Config1', w=1.0, fc=None, fe=None,
                 deadline=15.0, sla_penalty_scale=0.05, sla_lambda=1.0, return_vec=True,
                 task_size_cap=None,
                 perturb=False, spike_prob=0.06, spike_mult=5.0, spike_len=5):
        # perturb: 系统侧扰动 (测动作生成稳定性, 不含 ω 漂移=那是 ω 泛化)
        #   信道波动: 每步重 roll Rayleigh; 负载尖峰: 概率触发突发任务流。
        self.perturb = perturb
        self.spike_prob = spike_prob
        self.spike_mult = spike_mult
        self.spike_len = spike_len
        self._spike_left = 0
        # task_size_cap: 覆盖配置里的 task_size_H (任务大小上限)。
        #   multi-part 原版 100e6 bit / 边缘 2e6 bit/s = 单任务最多 50s 计算 -> 大任务必违约的结构地板。
        #   封顶 (如 20e6) 消除该地板, 让可行域能容下 delay-energy 权衡。
        # ===== SLA 扩展参数 (GMORL 原版没有) =====
        # deadline:           任务端到端时延截止期 (秒), 超过即违约
        #                     默认 15.0 ≈ multi-part 配置下随机策略时延 p75 (冒烟标定, ~25% 违约)
        # sla_penalty_scale:  把 "超时秒数" 折算成惩罚的系数 (与 r_T 量级对齐, 待冒烟标定)
        # sla_lambda:         标量奖励里 SLA 通道的权重 (与 ω 无关 —— 修掉 "SLA-进-r_T" 硬伤的关键)
        # return_vec:         step 的 info 是否带 3 通道向量 [r_T, r_E, r_C]
        self.deadline = deadline
        self.sla_penalty_scale = sla_penalty_scale
        self.sla_lambda = sla_lambda
        self.return_vec = return_vec
        self._task_id_counter = 0
        #读配置文件
        config = json.load(open(conf_file, 'r'))
        param = config[conf_name]
        self.dt = param['dt']
        self.Tmax = param['Tmax']
        self.edge_num_L = param['edge_num_L']
        self.edge_num_H = param['edge_num_H']
        self.user_num = param['user_num']
        self.possion_lamda = param['possion_lamda']
        
        self.task_size_L = param['task_size_L']
        self.task_size_H = param['task_size_H'] if task_size_cap is None else task_size_cap
        self.wave_cycle = param['wave_cycle']
        self.wave_peak = param['wave_peak']
        
        self.cloud_freq = param['cloud_cpu_freq']
        self.edge_freq = param['edge_cpu_freq']
        self.cloud_cpu_freq_peak = param['cloud_cpu_freq_peak']
        self.edge_cpu_freq_peak = param['edge_cpu_freq_peak']
        
            
        # if fc:
        #     self.cloud_cpu_freq = fc
        # if fe:
        #     for i in range(self.edge_num):
        #         self.edge_cpu_freq[i] = fe
        
        self.cloud_C = param['cloud_C']
        self.edge_C = param['edge_C']
        self.cloud_k = param['cloud_k']
        self.edge_k = param['edge_k']
        self.cloud_off_power = param['cloud_off_power']
        self.edge_off_power = param['edge_off_power']
        
        self.cloud_user_dist_H = param['cloud_user_dist_H']
        self.cloud_user_dist_L = param['cloud_user_dist_L']
        self.edge_user_dist_H = param['edge_user_dist_H']
        self.edge_user_dist_L = param['edge_user_dist_L']
        
        self.cloud_off_band_width = param['cloud_off_band_width']
        self.edge_off_band_width = param['edge_off_band_width']
        self.noise_dBm = param['noise_dBm']
        self.reward_alpha = param['reward_alpha']
        self.w = w
        
        self.reset()
        
    
    def reset(self):
        self.step_cnt = 0
        self.task_size = 0
        self.task_user_id = 0
        self.step_cloud_dtime = 0
        self.step_edge_dtime = 0
        self.step_energy = 0
        self.rew_t = 0
        self.rew_e = 0
        self.arrive_flag = False
        self.invalid_act_flag = False
        self.cloud_off_list = []
        self.cloud_exe_list = []
        self.edge_off_lists = []
        self.edge_exe_lists = []
        self.unassigned_task_list = []
        self.action = ACTION_TO_CLOUD
        # ===== SLA 记账 (每个 episode 重置) =====
        self._spike_left = 0
        self._task_id_counter = 0
        self._episode_delays = []      # 本 episode 所有已完成任务的端到端时延
        self._episode_energies = []    # 本 episode 所有已完成任务的能耗 (off+exe)
        self._episode_violations = 0   # 本 episode 累计违约数
        self._episode_finished_n = 0   # 本 episode 累计完成数

        self.edge_num = np.random.randint(self.edge_num_L, self.edge_num_H+1)
        self.action_space = self.edge_num + 1
        self.finish_time = np.array([0]*(self.edge_num+1))
        
        self.cloud_cpu_freq = np.random.uniform(self.cloud_freq-self.cloud_cpu_freq_peak, self.cloud_freq+self.cloud_cpu_freq_peak)
        self.edge_cpu_freq = [0]*self.edge_num
        self.task_size_exp_theta = self.cloud_cpu_freq/self.cloud_C
        for i in range(self.edge_num):
            self.edge_cpu_freq[i] = np.random.uniform(self.edge_freq-self.edge_cpu_freq_peak, self.edge_freq+self.edge_cpu_freq_peak)
            self.edge_off_lists.append([])
            self.edge_exe_lists.append([])
            self.task_size_exp_theta += self.edge_cpu_freq[i]/self.edge_C
        
        self.done = False
        self.reward_buff = []
        self.cloud_dist = np.random.uniform(self.cloud_user_dist_L, self.cloud_user_dist_H, size=(1, self.user_num))
        self.user_dist = self.cloud_dist
        for i in range(self.edge_num):
            edge_dist = np.random.uniform(self.edge_user_dist_L, self.edge_user_dist_H, size=(1, self.user_num))
            self.user_dist = np.concatenate((self.user_dist, edge_dist), axis=0)
            
        
        self.cloud_off_datarate, self.edge_off_datarate = self.updata_off_datarate()
        self.generate_task()
        return self.get_obs()
        
        
    def step(self, actions):
        assert self.done==False, 'enviroment already output done'
        self.step_cnt += 1
        # ===== 系统侧扰动 (perturb): 信道每步重 roll + 负载尖峰 =====
        if self.perturb:
            self.cloud_off_datarate, self.edge_off_datarate = self.updata_off_datarate()
            if self._spike_left > 0:
                self._spike_left -= 1
            elif np.random.rand() < self.spike_prob:
                self._spike_left = self.spike_len
        self.step_cloud_dtime = 0
        self.step_edge_dtime = 0
        self.step_energy = 0
        finished_task = []
        
        # self.invalid_act_flag = False
        #####################################################
        if self.arrive_flag:
            assert actions <= self.edge_num and actions >= ACTION_TO_CLOUD ,'action not in the interval %d, %d'%(actions,self.edge_num)

            self.action = actions
            self.arrive_flag = False
            the_task = {}
            the_task['tid'] = self._task_id_counter           # SLA: 任务唯一 id
            self._task_id_counter += 1
            the_task['start_step'] =  self.step_cnt           # = 分配步 (assign_step)
            the_task['user_id'] = self.task_user_id
            the_task['size'] = self.task_size
            the_task['remain'] = self.task_size
            the_task['off_time'] = 0
            the_task['wait_time'] = 0
            the_task['exe_time'] = 0
            the_task['off_energy'] = 0
            the_task['exe_energy'] = 0
            
            if actions == ACTION_TO_CLOUD:
                the_task['to'] = 0
                the_task['off_energy'] = (the_task['size']/self.cloud_off_datarate[the_task['user_id']])*self.cloud_off_power
                the_task['exe_energy'] = the_task['size']*self.cloud_k*self.cloud_C*(self.cloud_cpu_freq**2)
                self.step_energy = the_task['off_energy'] + the_task['exe_energy']
                self.cloud_off_list.append(the_task)
            else:
                e = actions
                the_task['to'] = e
                the_task['off_energy'] = (the_task['size']/self.edge_off_datarate[e-1, the_task['user_id']])*self.edge_off_power
                the_task['exe_energy'] = the_task['size']*self.edge_k*self.edge_C*(self.edge_cpu_freq[e-1]**2)
                self.step_energy = the_task['off_energy'] + the_task['exe_energy']
                self.edge_off_lists[e-1].append(the_task)
        self.rew_t, self.rew_e = self.estimate_rew()
                
        #####################################################
        self.generate_task()
        #####################################################
        used_time = 0
        while(used_time<self.dt):
            off_estimate_time = []
            exe_estimate_time = []
            task_off_num = len(self.cloud_off_list)
            task_exe_num = len(self.cloud_exe_list)

            for i in range(task_off_num):
                the_user = self.cloud_off_list[i]['user_id']
                estimate_time = self.cloud_off_list[i]['remain']/self.cloud_off_datarate[the_user]
                off_estimate_time.append(estimate_time)

            if task_exe_num > 0:
                cloud_exe_rate = self.cloud_cpu_freq/(self.cloud_C*task_exe_num)
            for i in range(task_exe_num):
                estimate_time = self.cloud_exe_list[i]['remain']/cloud_exe_rate
                exe_estimate_time.append(estimate_time)

            if len(off_estimate_time)+len(exe_estimate_time) > 0:
                min_time = min(off_estimate_time + exe_estimate_time)
            else:
                min_time = self.dt
   
            run_time = min(self.dt-used_time, min_time)

            cloud_pre_exe_list = []
            retain_flag_off = np.ones(task_off_num, dtype=np.bool_)
            for i in range(task_off_num):
                the_user = self.cloud_off_list[i]['user_id']
                self.cloud_off_list[i]['remain'] -= self.cloud_off_datarate[the_user]*run_time
                self.cloud_off_list[i]['off_energy'] += run_time*self.cloud_off_power

                self.cloud_off_list[i]['off_time'] += run_time
                if self.cloud_off_list[i]['remain'] <= ZERO_RES:
                    retain_flag_off[i] = False
                    the_task = deepcopy(self.cloud_off_list[i])
                    the_task['remain'] = self.cloud_off_list[i]['size']
                    cloud_pre_exe_list.append(the_task)
            pt = 0
            for i in range(task_off_num):
                if retain_flag_off[i]==False:
                    self.cloud_off_list.pop(pt)
                else:
                    pt += 1

            if task_exe_num > 0:
                cloud_exe_size = self.cloud_cpu_freq*run_time/(self.cloud_C*task_exe_num)
                cloud_exe_energy = self.cloud_k*run_time*(self.cloud_cpu_freq**3)/task_exe_num
            retain_flag_exe = np.ones(task_exe_num, dtype=np.bool_)
            for i in range(task_exe_num):
                self.cloud_exe_list[i]['remain'] -= cloud_exe_size
                self.cloud_exe_list[i]['exe_energy'] += cloud_exe_energy
                self.cloud_exe_list[i]['exe_time'] += run_time
                if self.cloud_exe_list[i]['remain'] <= ZERO_RES:
                    retain_flag_exe[i] = False
                    finished_task.append(self.cloud_exe_list[i])  # SLA: 收集完成任务
            pt = 0
            for i in range(task_exe_num):
                if retain_flag_exe[i]==False:
                    self.cloud_exe_list.pop(pt)
                else:
                    pt += 1
            self.cloud_exe_list = self.cloud_exe_list + cloud_pre_exe_list
            used_time += run_time
        #####################################################
        for n in range(self.edge_num):
            used_time = 0
            while(used_time<self.dt):
                off_estimate_time = []
                exe_estimate_time = []
                task_off_num = len(self.edge_off_lists[n])
                task_exe_num = len(self.edge_exe_lists[n])

                for i in range(task_off_num):
                    the_user = self.edge_off_lists[n][i]['user_id']
                    estimate_time = self.edge_off_lists[n][i]['remain']/self.edge_off_datarate[n,the_user]
                    off_estimate_time.append(estimate_time)

                if task_exe_num > 0:
                    edge_exe_rate = self.edge_cpu_freq[n]/(self.edge_C*task_exe_num)
                for i in range(task_exe_num):
                    estimate_time = self.edge_exe_lists[n][i]['remain']/edge_exe_rate
                    exe_estimate_time.append(estimate_time)

                if len(off_estimate_time)+len(exe_estimate_time) > 0:
                    min_time = min(off_estimate_time + exe_estimate_time)
                else:
                    min_time = self.dt

                run_time = min(self.dt-used_time, min_time)

                edge_pre_exe_list = []
                retain_flag_off = np.ones(task_off_num, dtype=np.bool_)
                for i in range(task_off_num):
                    the_user = self.edge_off_lists[n][i]['user_id']
                    self.edge_off_lists[n][i]['remain'] -= self.edge_off_datarate[n,the_user]*run_time
                    self.edge_off_lists[n][i]['off_energy'] += run_time*self.edge_off_power

                    self.edge_off_lists[n][i]['off_time'] += run_time
                    if self.edge_off_lists[n][i]['remain'] <= ZERO_RES:
                        retain_flag_off[i] = False
                        the_task = deepcopy(self.edge_off_lists[n][i])
                        the_task['remain'] = self.edge_off_lists[n][i]['size']
                        edge_pre_exe_list.append(the_task)
                pt = 0
                for i in range(task_off_num):
                    if retain_flag_off[i]==False:
                        self.edge_off_lists[n].pop(pt)
                    else:
                        pt += 1

                if task_exe_num > 0:
                    edge_exe_size = self.edge_cpu_freq[n]*run_time/(self.edge_C*task_exe_num)
                    edge_exe_energy = self.edge_k*run_time*(self.edge_cpu_freq[n]**3)/task_exe_num
                retain_flag_exe = np.ones(task_exe_num, dtype=np.bool_)
                for i in range(task_exe_num):
                    self.edge_exe_lists[n][i]['remain'] -= edge_exe_size
                    self.edge_exe_lists[n][i]['exe_energy'] += edge_exe_energy
                    self.edge_exe_lists[n][i]['exe_time'] += run_time
                    if self.edge_exe_lists[n][i]['remain'] <= ZERO_RES:
                        retain_flag_exe[i] = False
                        finished_task.append(self.edge_exe_lists[n][i])  # SLA: 收集完成任务
                pt = 0
                for i in range(task_exe_num):
                    if retain_flag_exe[i]==False:
                        self.edge_exe_lists[n].pop(pt)
                    else:
                        pt += 1
                self.edge_exe_lists[n] = self.edge_exe_lists[n] + edge_pre_exe_list
                used_time += run_time

        #####################################################
        if (self.step_cnt >= self.Tmax):
            self.done = True
        done = self.done

        #####################################################
        obs = self.get_obs()

        #####################################################
        # SLA: 从本步完成的任务里算第三通道 r_C (与 ω 无关) + 统计
        r_C, sla_stats = self._compute_sla(finished_task)
        r_T, r_E = self.rew_t, self.rew_e
        # 标量奖励 = GMORL 偏好加权 (r_T,r_E) + ω-无关的 SLA 惩罚 (sla_lambda*r_C)
        reward = self.w * r_T + (1.0 - self.w) * r_E + self.sla_lambda * r_C

        #####################################################
        info = {'sla': sla_stats}
        if self.return_vec:
            info['r_vec'] = np.array([r_T, r_E, r_C], dtype=np.float32)
            info['w'] = float(self.w)
        return obs, reward, done, info

    def _compute_sla(self, finished_task):
        """从本步完成的任务列表算 SLA 第三通道奖励 r_C 与统计。

        端到端时延 = off_time + exe_time (传输+计算的真实墙钟, 已含队列竞争拖慢;
        wait_time 在 GMORL 里恒为 0, v1 暂不单列, 见方案 §4.5)。
        r_C = -sla_penalty_scale * Σ max(0, delay - deadline)  —— 超时秒数惩罚, 与 ω 无关。

        注意 (方案已知项): 违约在"任务完成的那一步"结算, 惩罚落在当前步,
        而非造成违约的分配步 —— 存在信用分配滞后, v1 接受, v2 用 tid→assign_step 归因。
        """
        step_delays = []
        step_energies = []
        step_violations = 0
        excess_sum = 0.0
        for t in finished_task:
            delay = t['off_time'] + t['exe_time']
            step_delays.append(delay)
            step_energies.append(t['off_energy'] + t['exe_energy'])
            excess = delay - self.deadline
            if excess > 0:
                step_violations += 1
                excess_sum += excess
        r_C = -self.sla_penalty_scale * excess_sum
        # 累积到 episode 记账
        self._episode_delays.extend(step_delays)
        self._episode_energies.extend(step_energies)
        self._episode_violations += step_violations
        self._episode_finished_n += len(step_delays)
        sla_stats = {
            'n_finished': len(step_delays),
            'n_violated': step_violations,
            'excess_sum': excess_sum,
            'delays': step_delays,
        }
        return r_C, sla_stats

    def episode_sla_summary(self):
        """episode 结束时调用: 返回累计的 SLA 统计 (违约率 / 均值 / p95)。"""
        d = np.array(self._episode_delays, dtype=np.float64) if self._episode_delays else np.array([0.0])
        e = np.array(self._episode_energies, dtype=np.float64) if self._episode_energies else np.array([0.0])
        n = self._episode_finished_n
        return {
            'n_finished': n,
            'violation_rate': (self._episode_violations / n) if n > 0 else 0.0,
            'mean_delay': float(d.mean()),
            'mean_energy': float(e.mean()),
            'p95_delay': float(np.percentile(d, 95)),
            'p99_delay': float(np.percentile(d, 99)),
            'max_delay': float(d.max()),
        }

    # ===== SLA 准入掩码 (硬机制): 预测每服务器完成时间, 屏蔽预计超时的服务器 =====
    def predict_completion_times(self):
        """对当前到达任务 (self.task_size), 估计分配到每个合法服务器的端到端完成时延。

        估计 = 卸载时间 + (本任务 + 该服务器执行队列剩余总量) / 计算速率。
        处理器共享下这是个保守(偏大)估计 -> 准入会更谨慎, 对 SLA 有利。
        返回 [MAX_EDGE_NUM+1], 无效/补零槽 = inf。
        """
        ct = np.full(MAX_EDGE_NUM + 1, np.inf, dtype=np.float64)
        if self.task_size <= 0:           # 本步无真实任务 -> 都算"可行"
            ct[:self.edge_num + 1] = 0.0
            return ct
        for a in range(self.edge_num + 1):
            if a == ACTION_TO_CLOUD:
                datarate = self.cloud_off_datarate[self.task_user_id]
                rate = self.cloud_cpu_freq / self.cloud_C
                qwork = sum(t['remain'] for t in self.cloud_exe_list)
            else:
                datarate = self.edge_off_datarate[a - 1, self.task_user_id]
                rate = self.edge_cpu_freq[a - 1] / self.edge_C
                qwork = sum(t['remain'] for t in self.edge_exe_lists[a - 1])
            t_off = self.task_size / datarate if datarate > 0 else np.inf
            ct[a] = t_off + (self.task_size + qwork) / rate
        return ct

    def predict_energies(self):
        """对当前任务, 估计分配到每个合法服务器的能耗 (off+exe)。无效槽=inf。供 greedy-ω baseline。"""
        en = np.full(MAX_EDGE_NUM + 1, np.inf, dtype=np.float64)
        if self.task_size <= 0:
            en[:self.edge_num + 1] = 0.0
            return en
        for a in range(self.edge_num + 1):
            if a == ACTION_TO_CLOUD:
                datarate = self.cloud_off_datarate[self.task_user_id]
                off_e = (self.task_size / datarate) * self.cloud_off_power if datarate > 0 else np.inf
                exe_e = self.task_size * self.cloud_k * self.cloud_C * (self.cloud_cpu_freq ** 2)
            else:
                datarate = self.edge_off_datarate[a - 1, self.task_user_id]
                off_e = (self.task_size / datarate) * self.edge_off_power if datarate > 0 else np.inf
                exe_e = self.task_size * self.edge_k * self.edge_C * (self.edge_cpu_freq[a - 1] ** 2)
            en[a] = off_e + exe_e
        return en

    def admission_mask(self, margin=1.0):
        """[MAX_EDGE_NUM+1] 0/1: 预计能在 deadline*margin 内完成的合法服务器=1。
        margin<1 = 留安全裕度 (估计偏乐观时, 更严的准入); 若全不可行保留最快的那个。"""
        ct = self.predict_completion_times()
        m = (ct <= self.deadline * margin).astype(np.float32)
        if m.sum() < 0.5:
            m = np.zeros_like(m)
            m[int(np.argmin(ct))] = 1.0
        return m
    
    def generate_task(self):
        #####################################################
        _lam = self.possion_lamda * (self.spike_mult if self._spike_left > 0 else 1.0)  # 负载尖峰
        for u in range(self.user_num):
            task_num = np.random.poisson(_lam)
            for i in range(task_num):
                task = {}
                theta = self.task_size_exp_theta + self.wave_peak*np.sin(self.step_cnt*2*np.pi/self.wave_cycle)
                task_size = np.random.exponential(theta)
                task['task_size'] = np.clip(task_size, self.task_size_L, self.task_size_H)
                task['task_user_id'] = u
                self.unassigned_task_list.append(task)
            
        if self.step_cnt <= self.Tmax:
            if len(self.unassigned_task_list) > 0:
                self.arrive_flag = True
                arrive_task = self.unassigned_task_list.pop(0)
                self.task_size = arrive_task['task_size']
                self.task_user_id = arrive_task['task_user_id']
            else:
                self.arrive_flag = True
                self.task_size = 0
                self.task_user_id = np.random.randint(0, self.user_num)
            
            
    def updata_off_datarate(self):
        rayleigh = RAYLEIGH_VAR/2*(np.random.randn(self.edge_num+1, self.user_num)**2 + np.random.randn(self.edge_num+1, self.user_num)**2)  
        path_loss_dB = RAYLEIGH_PATH_LOSS_A*np.log10(self.user_dist/1000) + RAYLEIGH_PATH_LOSS_B
        total_path_loss_IndB = RAYLEIGH_ANTENNA_GAIN - RAYLEIGH_SHADOW_FADING - path_loss_dB
        path_loss = 10**(total_path_loss_IndB/10)
        rayleigh_noise_cloud = 10**((RAYLEIGH_NOISE_dBm-30)/10)*self.cloud_off_band_width;
        rayleigh_noise_edge = 10**((RAYLEIGH_NOISE_dBm-30)/10)*self.edge_off_band_width;
        gain_ = (path_loss*rayleigh)
        cloud_gain = gain_[0,:]/rayleigh_noise_cloud
        edge_gain = gain_[1:,:]/rayleigh_noise_edge
        cloud_noise = 10**((self.noise_dBm-30)/10)*self.cloud_off_band_width;
        edge_noise = 10**((self.noise_dBm-30)/10)*self.edge_off_band_width;
        cloud_off_datarate = self.cloud_off_band_width*np.log2(1 + (self.cloud_off_power*(cloud_gain**2))/cloud_noise)  
        edge_off_datarate = self.edge_off_band_width*np.log2(1 + (self.edge_off_power*(edge_gain**2))/edge_noise)  
        return cloud_off_datarate, edge_off_datarate

    
    def get_obs(self):
        obs = {}
        preference = [self.w, 1.0-self.w]
        freq = [self.cloud_cpu_freq/1e9] + [f/1e9 for f in self.edge_cpu_freq] + [0]*(MAX_EDGE_NUM-self.edge_num)
        obs['preference'] = np.array(preference)
        obs['frequency'] = np.array(freq)
        
        servers = []
        cloud = []
        cloud.append(1)
        cloud.append(self.cloud_cpu_freq/1e9)
        cloud.append(self.edge_num)
        cloud.append(self.task_size/1e6)
        cloud.append(1-self.done)
        cloud.append(self.cloud_off_datarate[self.task_user_id]/1e6/100)
        cloud.append(len(self.cloud_exe_list))
        task_exe_hist = np.zeros([60])
        n = 0
        for task in self.cloud_exe_list:
            task_feature = int(task['remain']/1e6)
            if task_feature>=60:
                task_feature = 59
            task_exe_hist[task_feature] += 1
        cloud = np.concatenate([np.array(cloud), task_exe_hist], axis=0)
        servers.append(cloud)
        
        for ii in range(self.edge_num):
            edge = []
            edge.append(1)
            edge.append(self.edge_cpu_freq[ii]/1e9)
            edge.append(self.edge_num)
            edge.append(self.task_size/1e6)
            edge.append(1-self.done)
            edge.append(self.edge_off_datarate[ii,self.task_user_id]/1e6/100)
            edge.append(len(self.edge_exe_lists[ii]))
            task_exe_hist = np.zeros([60])
            n = 0
            for task in self.edge_exe_lists[ii]:
                task_feature = int(task['remain']/1e6)
                if task_feature>=60:
                    task_feature = 59
                task_exe_hist[task_feature] += 1
            edge = np.concatenate([np.array(edge), task_exe_hist], axis=0)
            servers.append(edge)
            
        for ii in range(self.edge_num+1, MAX_EDGE_NUM+1):
            edge = -np.ones([67])
            servers.append(edge)
        
        obs['servers'] = np.array(servers).swapaxes(0,1)
        mask_a = np.array([[1]*(self.edge_num+1)+[0]*(MAX_EDGE_NUM-self.edge_num)]).T
        mask_b = np.array([[1]*(self.edge_num+1)+[0]*(MAX_EDGE_NUM-self.edge_num)])
        mask = np.matmul(mask_a, mask_b)
        obs['mask1'] = mask
        obs['mask2'] = np.array([1]*(self.edge_num+1)+[0]*(MAX_EDGE_NUM-self.edge_num))
        
        
        re = obs
        return re
    
    def estimate_rew(self):
        remain_list = []
        if self.action == ACTION_TO_CLOUD:
            for task in self.cloud_exe_list:
                remain_list.append(task['remain'])
            computing_speed = self.cloud_cpu_freq/self.cloud_C
            offload_time = self.task_size/self.cloud_off_datarate[self.task_user_id] if self.task_size>0 else 0
        else:
            for task in self.edge_exe_lists[self.action-1]:
                remain_list.append(task['remain'])
            computing_speed = self.edge_cpu_freq[self.action-1]/self.edge_C
            offload_time = self.task_size/self.edge_off_datarate[self.action-1][self.task_user_id] if self.task_size>0 else 0

        remain_list = np.sort(remain_list)
        
        last_size = 0
        t2 = 0
        task_num = len(remain_list)
        for i in range(task_num):
            size = remain_list[i]
            current_speed = computing_speed/(task_num-i)
            t2 += (task_num-i)*(size-last_size)/current_speed
            last_size = size
        
        last_size = 0
        t_norm = 0
        t1 = 0
        task_num = len(remain_list)
        for i in range(task_num):
            size = remain_list[i]
            current_speed = computing_speed/(task_num-i)
            use_t = (size-last_size)/current_speed
            if t_norm + use_t >= offload_time:
                t_cut = offload_time - t_norm
                t1 += (task_num-i)*t_cut
                t_norm = offload_time
                remain_list[i] -= t_cut*current_speed
                remain_list[i] = 0 if remain_list[i]<ZERO_RES else remain_list[i]
                remain_list = remain_list[i:]
                break
            else:
                t1 += (task_num-i)*(size-last_size)/current_speed
                t_norm += use_t
            last_size = size
        
        remain_list = remain_list.tolist()
        remain_list.append(self.task_size)
        remain_list = np.sort(remain_list)
        last_size = 0
        task_num = len(remain_list)
        for i in range(task_num):
            size = remain_list[i]
            current_speed = computing_speed/(task_num-i)
            t1 += (task_num-i)*(size-last_size)/current_speed
            last_size = size
        
        # reward_dt = t2 - t1  
        reward_dt = t1 - t2
        if self.task_size > 0:
            reward_dt = -reward_dt*0.01
            reward_de = -self.step_energy*5
        else:
            reward_dt = 0
            reward_de = 0
        
        return reward_dt, reward_de
    
    def get_reward(self, finished_task):

        reward = self.w*self.rew_t + (1.0-self.w)*self.rew_e
        
        return reward

    
    def rander(self):
        pass
    