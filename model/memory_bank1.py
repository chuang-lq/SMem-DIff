import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from einops import rearrange


def softmax_w_top(x, top):
    top = min(top, x.shape[1])
    values, indices = torch.topk(x, k=top, dim=1)
    x_exp = values.exp_()
    x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
    # x.zero_().scatter_(1, indices, x_exp)
    result = torch.zeros_like(x).scatter(1, indices, x_exp)

    return result


class MemoryBank:
    def __init__(self, para, count_usage: bool, top_k: Optional[int]=None):
        self.count_usage = count_usage
        self.top_k = top_k

        self.CK = None
        self.CV = None
        self.HW = None
        self.mem_cnt = 0
        self.mem_every = para.mem_every

        self.mem_k = None
        self.mem_v = None

        # shrinkage and selection are also single tensors
        self.s = None
        self.e = None

        # usage
        if self.count_usage:
            self.use_count = self.life_count = None
            self.frame_times = None

        # The hidden state will be stored in a single tensor
        self.hidden_dim = 96
        self.hidden = None

    def _global_matching(self, mk, ms, qk, qe, return_usage=False):
        # mk: B x CK x [N]    - Memory keys
        # ms: B x  1 x [N]    - Memory shrinkage
        # qk: B x CK x [HW/P] - Query keys
        # qe: B x CK x [HW/P] - Query selection

        B, CK, NE = mk.shape

        if qe is not None:
            # (a-b)^2 = a^2 - 2ab + b^2
            mk = mk.transpose(1, 2)
            a_sq = (mk.pow(2) @ qe)
            two_ab = 2 * (mk @ (qk * qe))
            b_sq = (qe * qk.pow(2)).sum(1, keepdim=True)
            similarity = (-a_sq + two_ab - b_sq)
        else:
            # similar to STCN if we don't have the selection term
            a_sq = mk.pow(2).sum(1).unsqueeze(2)
            two_ab = 2 * (mk.transpose(1, 2) @ qk)
            similarity = (-a_sq + two_ab)

        if ms is not None:
            ms = ms.flatten(start_dim=1).unsqueeze(2)  # B, THW, 1
            similarity = similarity * ms / math.sqrt(CK)  # B, THW, HW
        else:
            similarity = similarity / math.sqrt(CK)

        if self.top_k is None:
            maxes = torch.max(similarity, dim=1, keepdim=True)[0]
            x_exp = torch.exp(similarity - maxes)
            x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
            affinity = x_exp / x_exp_sum
        else:
            affinity = softmax_w_top(similarity, top=self.top_k)  # B, THW, HW

        if return_usage:
            return affinity, affinity.sum(dim=-1)

        return affinity

    def _readout(self, affinity, mv):
        # mv: B, CV, THW
        return torch.bmm(mv, affinity)  # B, CV, HW

    def match_memory(self, qk, qe):
        b, c_k, h, w = qk.shape

        qk = qk.flatten(start_dim=2)
        if qe is not None:
            qe = qe.flatten(start_dim=2)

        mk = self.mem_k
        mv = self.mem_v
        ms = self.s

        if self.count_usage:
            affinity, usage = self._global_matching(mk, ms, qk, qe, return_usage=True)
            self.update_usage(usage)
        else:
            affinity = self._global_matching(mk, ms, qk, qe)

        readout_mem = self._readout(affinity, mv)

        return readout_mem.view(b, self.CV, h, w)

    def add_memory(self, key, value, shrinkage, selection=None):

        key = key.flatten(start_dim=2)
        value = value.flatten(start_dim=2)
        if shrinkage is not None:
            shrinkage = shrinkage.flatten(start_dim=2)
        if selection is not None:
            selection = selection.flatten(start_dim=2)

        b = key.shape[0]
        new_count = torch.zeros((b, 1, key.shape[2]), device=key.device, dtype=torch.float32)
        new_life = torch.zeros((b, 1, key.shape[2]), device=key.device, dtype=torch.float32) + 1e-7

        if self.mem_k is None:
            self.mem_k = key
            self.mem_v = value
            self.s = shrinkage
            self.e = selection
            if self.count_usage:
                self.use_count = new_count
                self.life_count = new_life
                # self.frame_times = [[1] for i in range(b)]
            self.HW = key.shape[-1]
            self.CK = key.shape[1]
            self.CV = value.shape[1]
        else:
            if self.mem_cnt == self.mem_every:
                self.remove_obsolete_feature(n=4)
                # self.update_memory()

            self.mem_k = torch.cat([self.mem_k, key], -1)
            self.mem_v = torch.cat([self.mem_v, value], -1)
            if shrinkage is not None:
                self.s = torch.cat([self.s, shrinkage], -1)
            if selection is not None:
                self.e = torch.cat([self.e, selection], -1)
            if self.count_usage:
                self.use_count = torch.cat([self.use_count, new_count], -1)
                self.life_count = torch.cat([self.life_count, new_life], -1)
                # for i in range(b):
                #     self.frame_times[i] = [j + 1 for j in self.frame_times[i]]
                #     self.frame_times[i].append(1)

        self.mem_cnt += 1

    def create_hidden_state(self, sample_key):
        b, _, h, w = sample_key.shape
        if self.hidden is None:
            self.hidden = torch.zeros((b, self.hidden_dim, h, w), device=sample_key.device, dtype=torch.float32)

    def set_hidden(self, hidden_state):
        self.hidden = hidden_state

    def get_hidden(self):
        return self.hidden

    def update_usage(self, usage):
        # increase all life count by 1
        # increase use of indexed elements
        if not self.count_usage:
            return

        self.use_count += usage.view_as(self.use_count)
        self.life_count += 1

    def get_usage(self):
        # return normalized usage
        if not self.count_usage:
            raise RuntimeError('I did not count usage!')
        else:
            usage = self.use_count / self.life_count
            return usage

    def remove_obsolete_feature(self, n=1):
        # normalize with life duration
        usage = self.get_usage().flatten(start_dim=1)
        b = usage.shape[0]
        _, indices = torch.topk(usage, k=(self.mem_every - n) * self.HW, dim=-1, largest=True, sorted=True)

        mk = torch.zeros((b, self.CK, (self.mem_every - n) * self.HW), device=usage.device, dtype=torch.float32)
        mv = torch.zeros((b, self.CV, (self.mem_every - n) * self.HW), device=usage.device, dtype=torch.float32)
        if self.s is not None:
            ms = torch.zeros((b, 1, (self.mem_every - n) * self.HW), device=usage.device, dtype=torch.float32)
        if self.e is not None:
            me = torch.zeros((b, self.CK, (self.mem_every - n) * self.HW), device=usage.device, dtype=torch.float32)
        use_c = torch.zeros((b, 1, (self.mem_every - n) * self.HW), device=usage.device, dtype=torch.float32)
        life_c = torch.zeros((b, 1, (self.mem_every - n) * self.HW), device=usage.device, dtype=torch.float32)

        for i in range(b):
            indice = indices[i]
            mk[i] = self.mem_k[i, :, indice]
            mv[i] = self.mem_v[i, :, indice]
            if self.s is not None:
                ms[i] = self.s[i, :, indice]
            if self.e is not None:
                me[i] = self.e[i, :, indice]
            use_c[i] = self.use_count[i, :, indice]
            life_c[i] = self.life_count[i, :, indice]

        self.mem_k = mk
        self.mem_v = mv
        self.s = ms if self.s is not None else None
        self.e = me if self.e is not None else None
        self.use_count = use_c
        self.life_count = life_c

        self.mem_cnt -= n

    def remove_obsolete_feature2(self):
        # normalize with life duration
        usage = self.get_usage().flatten(start_dim=1)
        b = usage.shape[0]
        usage = usage.view(b, self.mem_cnt, self.HW).sum(dim=-1)
        usage = usage / usage.sum(dim=-1, keepdim=True)
        to_drop_idx = torch.argmin(usage, dim=-1, keepdim=True)  # (b, 1)

        mk = torch.zeros((b, self.CK, 7 * self.HW), device=usage.device, dtype=torch.float32)
        mv = torch.zeros((b, self.CV, 7 * self.HW), device=usage.device, dtype=torch.float32)
        if self.s is not None:
            ms = torch.zeros((b, 1, 7 * self.HW), device=usage.device, dtype=torch.float32)
        if self.e is not None:
            qe = torch.zeros((b, self.CK, 7 * self.HW), device=usage.device, dtype=torch.float32)
        use_c = torch.zeros((b, 1, 7 * self.HW), device=usage.device, dtype=torch.float32)
        life_c = torch.zeros((b, 1, 7 * self.HW), device=usage.device, dtype=torch.float32)

        for i in range(b):
            indice = to_drop_idx[i]
            mk[i] = torch.cat([self.mem_k[i, :, 0:indice * self.HW], self.mem_k[i, :, (indice + 1) * self.HW:]], dim=-1)
            mv[i] = torch.cat([self.mem_v[i, :, 0:indice * self.HW], self.mem_v[i, :, (indice + 1) * self.HW:]], dim=-1)
            if self.s is not None:
                ms[i] = torch.cat([self.s[i, :, 0:indice * self.HW], self.s[i, :, (indice + 1) * self.HW:]], dim=-1)
            if self.e is not None:
                qe[i] = torch.cat([self.e[i, :, 0:indice * self.HW], self.e[i, :, (indice + 1) * self.HW:]], dim=-1)
            use_c[i] = torch.cat(
                [self.use_count[i, :, 0:indice * self.HW], self.use_count[i, :, (indice + 1) * self.HW:]], dim=-1)
            life_c[i] = torch.cat(
                [self.life_count[i, :, 0:indice * self.HW], self.life_count[i, :, (indice + 1) * self.HW:]], dim=-1)

        self.mem_k = mk
        self.mem_v = mv
        self.s = ms if self.s is not None else None
        self.e = qe if self.e is not None else None
        self.use_count = use_c
        self.life_count = life_c

        self.mem_cnt -= 1

    def update_memory(self):
        usage = self.get_usage().flatten(start_dim=1)
        b = usage.shape[0]
        usage = usage.view(b, self.mem_cnt, self.HW).sum(dim=-1)
        usage = usage / usage.sum(dim=-1, keepdim=True)
        frame_times = torch.tensor(self.frame_times, device=usage.device)
        # UCB(Relevance + Freshness): Rk + sqrt(2logT/tk)
        add_item = 8
        mul_item = 1.5
        frame_times_para = mul_item * torch.sqrt(torch.log(frame_times.sum(dim=-1, keepdim=True)) / (frame_times + add_item))
        attn_weight = usage + frame_times_para
        to_drop_idx = torch.argmin(attn_weight, dim=-1, keepdim=True)  # (b, 1)

        mk = torch.zeros((b, self.CK, 7 * self.HW), device=usage.device, dtype=torch.float32)
        mv = torch.zeros((b, self.CV, 7 * self.HW), device=usage.device, dtype=torch.float32)
        if self.s is not None:
            ms = torch.zeros((b, 1, 7 * self.HW), device=usage.device, dtype=torch.float32)
        if self.e is not None:
            qe = torch.zeros((b, self.CK, 7 * self.HW), device=usage.device, dtype=torch.float32)
        use_c = torch.zeros((b, 1, 7 * self.HW), device=usage.device, dtype=torch.float32)
        life_c = torch.zeros((b, 1, 7 * self.HW), device=usage.device, dtype=torch.float32)

        for i in range(b):
            indice = to_drop_idx[i]
            mk[i] = torch.cat([self.mem_k[i, :, 0:indice * self.HW], self.mem_k[i, :, (indice+1) * self.HW:]], dim=-1)
            mv[i] = torch.cat([self.mem_v[i, :, 0:indice * self.HW], self.mem_v[i, :, (indice+1) * self.HW:]], dim=-1)
            if self.s is not None:
                ms[i] = torch.cat([self.s[i, :, 0:indice * self.HW], self.s[i, :, (indice+1) * self.HW:]], dim=-1)
            if self.e is not None:
                qe[i] = torch.cat([self.e[i, :, 0:indice * self.HW], self.e[i, :, (indice+1) * self.HW:]], dim=-1)
            use_c[i] = torch.cat([self.use_count[i, :, 0:indice * self.HW], self.use_count[i, :, (indice+1) * self.HW:]], dim=-1)
            life_c[i] = torch.cat([self.life_count[i, :, 0:indice * self.HW], self.life_count[i, :, (indice+1) * self.HW:]], dim=-1)
            self.frame_times[i].pop(indice)

        self.mem_k = mk
        self.mem_v = mv
        self.s = ms if self.s is not None else None
        self.e = qe if self.e is not None else None
        self.use_count = use_c
        self.life_count = life_c

        self.mem_cnt -= 1

    def init_memory(self):
        if self.mem_k is not None:
            self.mem_k = self.mem_k.detach()
            self.mem_v = self.mem_v.detach()
            if self.s is not None:
                self.s = self.s.detach()
            if self.e is not None:
                self.e = self.e.detach()
            if self.hidden is not None:
                self.hidden = self.get_hidden().detach()
            if self.count_usage:
                self.use_count = self.use_count.detach()
                self.life_count = self.life_count.detach()

    def clear_memory(self):
        self.mem_k = None
        self.mem_v = None
        self.s = None
        self.e = None
        self.mem_cnt = 0
        if self.hidden is not None:
            self.hidden = self.get_hidden().detach()
        if self.count_usage:
            self.use_count = None
            self.life_count = None
            self.frame_times = None


# if __name__ == '__main__':
#     x = torch.randn(2, 3, 64)
#     y = torch.randn(2, 3, 64)
#     o = torch.cat([x, y], dim=-1)
#     print(o.shape)
#     o1 = o.view(2, 3, 2, 64)
#     # print(o1[:, :, 0] == x)  # true
#     o2 = o1.permute(0, 3, 2, 1)
#     times = [[1] for i in range(2)]
#     for i in range(2):
#         times[i] = [j+1 for j in times[i]]
#         times[i].append(1)
#     times = torch.tensor(times)
#
#     print(times)
#     attn = torch.sqrt(torch.log(times.sum(dim=-1, keepdim=True)) / (times + 3))
#     print(attn)
#     attn = torch.randn(2, 7)
#     times = [1, 2, 3, 4, 5, 6]
#     times = [i+1 for i in times]
#     times.append(1)
#     print(times)
#     times.pop(5)
#     times = [i + 1 for i in times]
#     times.append(1)
#     times = torch.tensor(times)
#     attn = attn + times
#     to_drop_idx = torch.argmin(attn, dim=-1, keepdim=True)
#     print(times)
#     print(to_drop_idx.shape)
