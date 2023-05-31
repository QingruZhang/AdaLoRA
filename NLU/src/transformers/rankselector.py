import os
import sys 
import argparse
import logging
import random
import torch
import numpy as np


class RankSelector(object):
    def __init__(self, model, args, total_step=None, tb_writter=None,):
        self.model = model
        self.config = vars(args)
        self.args = args

        self.ipt = {} 
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}
        self.cat_ipt = {}

        self.ave_target_rank = args.target_rank 
        self.target_rank = args.target_total_rank
        self.lora_init_rank = args.lora_r 
        self.initial_warmup = args.init_warmup
        self.final_warmup = args.final_warmup 
        self.total_step = total_step
        self.mask_interval = args.mask_interval
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.logipt = args.logipt
        if args.combine_ipt_fun == "mean": 
            self.combine_func = torch.mean 
        elif args.combine_ipt_fun == "sum": 
            self.combine_func = torch.sum
        self.finalize_rank = args.finalize_rank 
        self.adapt_scaling = args.adapt_scaling 

        self.tb_writter = tb_writter
        self.log_interval = args.tb_writter_loginterval 
        self.plot_cov = True

        self.rank_pattern = {}

        self.get_lora_param_name()
        ####### 
        # self.att_enable_lora = args.att_enable_lora 

    def set_total_step(self, total_step):
        self.total_step = total_step

    def get_rank_pattern(self):
        return self.rank_pattern 

    def get_lora_param_name(self):
        self.name_set = set() 
        self.total_rank = 0 
        self.shape_dict = {}
        for n,p in self.model.named_parameters():
            if "lora_A" in n: 
                name_mat = n.replace("lora_A", "%s")
                self.name_set.add(name_mat)
                self.total_rank += p.size(0) 
                self.shape_dict[n] = p.shape
            if "lora_B" in n:
                self.shape_dict[n] = p.shape
        self.name_set = list(sorted(self.name_set)) 
        if self.target_rank is None:
            self.target_rank = self.ave_target_rank * len(self.name_set) 


    def schedule_threshold(self, step:int,):
        mask_ind = False 
        target_rank = self.target_rank 
        initial_warmup = self.initial_warmup 
        final_warmup = self.final_warmup 
        total_step = self.total_step 
        self.global_step = step
        if step <= initial_warmup: 
            curr_rank = self.total_rank 
            mask_ind = False 
        elif step > total_step - final_warmup: 
            curr_rank = self.target_rank 
            mask_ind = True 
        else: 
            mul_coeff = 1-(step-initial_warmup)/(total_step-final_warmup-initial_warmup)
            curr_rank = target_rank + (self.total_rank-target_rank)*(mul_coeff**3)
            curr_rank = int(curr_rank)
            mask_ind = True if step % self.mask_interval == 0 else False 
        return curr_rank, mask_ind 


    def update_ipt(self, model): 
        for n,p in model.named_parameters():
            if "lora_" in n: 
                if n not in self.ipt:
                    self.ipt[n] = torch.zeros_like(p)
                    self.exp_avg_ipt[n] = torch.zeros_like(p) 
                    self.exp_avg_unc[n] = torch.zeros_like(p) 
                # print("p device", p.device, n)
                # print("ipt device", self.ipt[n].device, n)
                with torch.no_grad():
                    if self.logipt:
                        self.ipt[n] = (p * p.grad+1e-50).abs().log().detach()
                    else:
                        self.ipt[n] = (p * p.grad).abs().detach()
                    self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + (1-self.beta1)*self.ipt[n]
                    if self.beta2>0 and self.beta2 <1:
                        self.exp_avg_unc[n] = self.beta2 * self.exp_avg_unc[n] + (1-self.beta2)*(self.ipt[n]-self.exp_avg_ipt[n]).abs()

    def calculate_score(self, n):
        if self.beta2 <=0 or self.beta2 >= 1:
            ipt_score = self.exp_avg_ipt[n]
        elif self.logipt: 
            ipt_score = self.exp_avg_ipt[n] + self.exp_avg_unc[n] 
        else:
            ipt_score = self.exp_avg_ipt[n] * self.exp_avg_unc[n]
        return ipt_score

    def mask_to_target_rank(self, model, curr_rank): 
        is_dict = {}
        combine_dict = {} 
        for n,p in model.named_parameters(): 
            if "lora_A" in n: 
                rdim, hdim_a = p.shape
                ipt_score = self.calculate_score(n)
                comb_ipt = self.combine_func(ipt_score, dim=1, keepdim=True)
                name_mat = n.replace("lora_A", "%s")
                if name_mat not in combine_dict: 
                    combine_dict[name_mat] = [comb_ipt]
                else:
                    combine_dict[name_mat].append(comb_ipt)
            if "lora_B" in n: 
                hdim_b, rdim = p.shape 
                ipt_score = self.calculate_score(n)
                comb_ipt = self.combine_func(ipt_score, dim=0, keepdim=False).view(-1, 1)
                name_mat = n.replace("lora_B", "%s")
                if name_mat not in combine_dict: 
                    combine_dict[name_mat] = [comb_ipt]
                else:
                    combine_dict[name_mat].append(comb_ipt)

        all_is = []
        for name_mat in combine_dict: 
            sum_ipt = torch.cat(combine_dict[name_mat], dim=1).sum(dim=1, keepdim=False)
            name_A = name_mat%"lora_A"
            name_B = name_mat%"lora_B"
            hdim_a = self.shape_dict[name_A][1]
            hdim_b = self.shape_dict[name_B][0]
            is_dict[name_A] = sum_ipt.view(-1, 1).repeat((1, hdim_a))
            is_dict[name_B] = sum_ipt.view(1, -1).repeat((hdim_b, 1))
            # is_dict[name_B] = sum_ipt.view(ndim_b, rdim, 1).repeat(1,1,hdim_b).permute(0,2,1).contiguous().view(ndim_b*hdim_b, rdim)
            all_is.append(sum_ipt.view(-1))
        # torch.use_deterministic_algorithms(False)
        mask_threshold = torch.kthvalue(torch.cat(all_is), (self.total_rank-curr_rank))[0].item()
        # torch.use_deterministic_algorithms(True) 

        with torch.no_grad():
            curr_sum_rank = 0
            sum_param = 0
            for n,p in model.named_parameters():
                if "lora_" in n: 
                    p.data.masked_fill_(is_dict[n]<=mask_threshold, 0.0)

                    if self.tb_writter is not None and "lora_A" in n and self.global_step % self.log_interval==0:
                        # sum_rank = (p[:, 0] != 0.0).view(-1).sum() 
                        sum_rank = (p.sum(dim=1).detach() == 0.0).sum().item() 
                        # sum_rank = (is_dict[n][:, 0] > mask_threshold).view(-1).sum() 
                        rec_name = n.split(".")[-5] + "_" + n.split(".")[-4] + "_" + n.split(".")[-2] + "_" + n.split(".")[-1]
                        # num_mat = sum(self.att_enable_lora)
                        # sum_rank = (is_dict[n][:, 0] > mask_threshold).view(num_mat, -1).sum(dim=1)
                        # for i,k in enumerate(self.att_enable_lora):
                        #     if k != 0:
                        #         self.tb_writter.add_scalar("Rank/%s/%d"%(rec_name, i+1), sum_rank[sum(self.att_enable_lora[:i+1])-1], self.global_step)
                        # self.tb_writter.add_scalar("Rank/%s/%d"%(n, 1), sum_rank[0], self.global_step) 
                        # self.tb_writter.add_scalar("Rank/%s/%d"%(n, 2), sum_rank[1], self.global_step) 
                        # self.tb_writter.add_scalar("Rank/%s/%d"%(n, 3), sum_rank[2], self.global_step) 
                        # self.tb_writter.add_scalar("Rank/%s"%(rec_name,), sum_rank, self.global_step) 
                        self.tb_writter.add_scalar("Rank/%s"%(n,), sum_rank, self.global_step) 
                        unmasked_rank = self.lora_init_rank-sum_rank
                        self.tb_writter.add_scalar("Unmask_Rank/%s"%(n,), self.lora_init_rank-sum_rank, self.global_step)
                        self.rank_pattern[n] = unmasked_rank 
                        curr_sum_rank += unmasked_rank 
                        sum_param += unmasked_rank*self.shape_dict[n][1]*2 

                        self.tb_writter.add_scalar("Ipt_Sc/Ipt/%s"%rec_name, is_dict[n][:, 0].sum().item(), self.global_step)
                        self.tb_writter.add_scalar("Ipt_Sc/Sen/%s"%rec_name, self.ipt[n].sum().item(), self.global_step)
                        self.tb_writter.add_scalar("Ipt_Sc/Unc/%s"%rec_name, self.exp_avg_unc[n].sum().item(), self.global_step)
                        name_b = n.replace("lora_A", "lora_B")
                        rec_name_b = rec_name.replace("lora_A", "lora_B")
                        self.tb_writter.add_scalar("Ipt_Sc/Sen/%s"%rec_name_b, self.ipt[name_b].sum().item(), self.global_step)
                        self.tb_writter.add_scalar("Ipt_Sc/Unc/%s"%rec_name_b, self.exp_avg_unc[name_b].sum().item(), self.global_step)

            if self.tb_writter is not None and self.global_step%self.log_interval==0:
                self.tb_writter.add_scalar("Prune/total_rank", curr_sum_rank, self.global_step)
                self.tb_writter.add_scalar("Prune/mask_threshold", mask_threshold, self.global_step)
                self.tb_writter.add_scalar("Prune/sum_param", sum_param, self.global_step)

        return mask_threshold


    def update_and_mask(self, model, global_step):
        if global_step<self.total_step-self.final_warmup or not self.finalize_rank:
            self.update_ipt(model)
        curr_rank, mask_ind = self.schedule_threshold(global_step)
        if mask_ind:
            mask_threshold = self.mask_to_target_rank(model, curr_rank) 
        else:
            mask_threshold = None 
        if self.plot_cov and self.tb_writter is not None and self.global_step%self.log_interval==0:
            with torch.no_grad():
                sum_coef = []
                for n,p in model.named_parameters():
                    if "lora_" in n:
                        mat = p.data.detach().clone()
                        mat_cov = mat @ mat.T if "lora_A" in n else mat.T @ mat 
                        norm_dim = 1 if "lora_A" in n else 0 
                        mat_norm = mat.norm(p="fro", dim=norm_dim) 
                        epsilon = 1e-30 
                        cov_coef = mat_cov.abs() / (mat_norm.view(1, -1) * mat_norm.view(-1, 1) + epsilon)
                        d = cov_coef.shape[0]
                        orth_coef = (cov_coef.sum() - cov_coef.trace())/(d*d-d) 
                        sum_coef.append(orth_coef.item())
                        self.tb_writter.add_scalar("Orth_Coef/%s"%n, orth_coef.item(), self.global_step)
                self.tb_writter.add_scalar("Orth_Coef/sum_orth", sum(sum_coef)/len(sum_coef), self.global_step)

        return curr_rank, mask_threshold



class SVDRankSelector(object):
    def __init__(self, model, args, total_step=None, tb_writter=None,):
        self.model = model
        self.config = vars(args)
        self.args = args

        self.ipt = {} 
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}
        self.cat_ipt = {}

        self.lora_type = args.lora_type 
        self.ave_target_rank = args.target_rank 
        self.target_rank = args.target_total_rank
        self.lora_init_rank = args.lora_r 
        self.select_metric = args.select_metric 
        self.initial_warmup = args.init_warmup
        self.final_warmup = args.final_warmup 
        self.total_step = total_step
        self.mask_interval = args.mask_interval
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.logipt = args.logipt
        if args.combine_ipt_fun == "mean": 
            self.combine_func = torch.mean 
        elif args.combine_ipt_fun == "sum": 
            self.combine_func = torch.sum
        self.finalize_rank = args.finalize_rank 
        self.adapt_scaling = args.adapt_scaling 

        self.select_metric = self.select_metric.split(",")
        assert len(self.select_metric)==4 

        self.tb_writter = tb_writter
        self.log_interval = args.tb_writter_loginterval 
        self.plot_cov = True

        self.rank_pattern = {} 

        self.get_lora_param_name()
        ####### 
        # self.att_enable_lora = args.att_enable_lora 

    def set_total_step(self, total_step):
        self.total_step = total_step

    def get_rank_pattern(self):
        return self.rank_pattern

    def get_lora_param_name(self):
        self.name_set = set() 
        self.total_rank = 0 
        self.shape_dict = {}
        for n,p in self.model.named_parameters():
            if "lora_A" in n: 
                name_mat = n.replace("lora_A", "%s")
                self.name_set.add(name_mat)
                self.total_rank += p.size(0) 
                self.shape_dict[n] = p.shape
            if "lora_B" in n:
                self.shape_dict[n] = p.shape
        self.name_set = list(sorted(self.name_set)) 
        if self.target_rank is None:
            self.target_rank = self.ave_target_rank * len(self.name_set) 


    def schedule_threshold(self, step:int,):
        mask_ind = False 
        target_rank = self.target_rank 
        initial_warmup = self.initial_warmup 
        final_warmup = self.final_warmup 
        total_step = self.total_step 
        self.global_step = step
        if step <= initial_warmup: 
            curr_rank = self.total_rank 
            mask_ind = False 
        elif step > total_step - final_warmup: 
            curr_rank = self.target_rank 
            mask_ind = True 
        else: 
            mul_coeff = 1-(step-initial_warmup)/(total_step-final_warmup-initial_warmup)
            curr_rank = target_rank + (self.total_rank-target_rank)*(mul_coeff**3)
            curr_rank = int(curr_rank)
            mask_ind = True if step % self.mask_interval == 0 else False 
        return curr_rank, mask_ind 


    def update_ipt(self, model): 
        for n,p in model.named_parameters():
            if "lora_" in n: 
                if n not in self.ipt:
                    self.ipt[n] = torch.zeros_like(p)
                    self.exp_avg_ipt[n] = torch.zeros_like(p) 
                    self.exp_avg_unc[n] = torch.zeros_like(p) 
                # print("p device", p.device, n)
                # print("ipt device", self.ipt[n].device, n)
                with torch.no_grad():
                    if self.logipt:
                        self.ipt[n] = (p * p.grad+1e-35).abs().log().detach()
                    else:
                        self.ipt[n] = (p * p.grad).abs().detach()
                    self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + (1-self.beta1)*self.ipt[n]
                    if self.beta2>0 and self.beta2 <1:
                        self.exp_avg_unc[n] = self.beta2 * self.exp_avg_unc[n] + (1-self.beta2)*(self.ipt[n]-self.exp_avg_ipt[n]).abs()

    def calculate_score(self, n, p=None, metric="iptAB"):
        if "ipt" in metric:
            if self.beta2 <=0 or self.beta2 >= 1:
                ipt_score = self.exp_avg_ipt[n]
            elif self.logipt: 
                ipt_score = self.exp_avg_ipt[n] + self.exp_avg_unc[n] 
            else:
                ipt_score = self.exp_avg_ipt[n] * self.exp_avg_unc[n]
        elif "mag" in metric:
            ipt_score = p.abs().detach().clone() 
        else:
            raise ValueError("Unexcptected Metric: %s"%metric)
        return ipt_score 

    def _combine_ipt(self, ipt_E, ipt_AB):
        operator_AB = self.select_metric[2]
        if operator_AB == "sumAB":
            ipt_AB = ipt_AB.sum(dim=1, keepdim=False)
        elif operator_AB == "prodAB":
            ipt_AB = ipt_AB.prod(dim=1, keepdim=False)
        else:
            raise ValueError("Unexcepted AB Operator: %s"%operator_AB)
        operator_E = self.select_metric[3]
        if operator_E == "sumE":
            sum_ipt = ipt_E.view(-1) + ipt_AB.view(-1)
        elif operator_E == "prodE":
            sum_ipt = ipt_E.view(-1) * ipt_AB.view(-1)
        elif operator_E == "onlyE":
            sum_ipt = ipt_E.view(-1) 
        else:
            raise ValueError("Unexcepted E Operator: %s"%operator_E)
        return sum_ipt


    def mask_to_target_rank(self, model, curr_rank): 
        is_dict = {}
        combine_dict = {} 
        singular_dict = {}
        for n,p in model.named_parameters(): 
            if "lora_A" in n: 
                rdim, hdim_a = p.shape
                ipt_score = self.calculate_score(n, metric=self.select_metric[0])
                comb_ipt = self.combine_func(ipt_score, dim=1, keepdim=True)
                name_mat = n.replace("lora_A", "%s")
                if name_mat not in combine_dict: 
                    combine_dict[name_mat] = [comb_ipt]
                else:
                    combine_dict[name_mat].append(comb_ipt)
            if "lora_B" in n: 
                hdim_b, rdim = p.shape 
                ipt_score = self.calculate_score(n, metric=self.select_metric[0])
                comb_ipt = self.combine_func(ipt_score, dim=0, keepdim=False).view(-1, 1)
                name_mat = n.replace("lora_B", "%s")
                if name_mat not in combine_dict: 
                    combine_dict[name_mat] = [comb_ipt]
                else:
                    combine_dict[name_mat].append(comb_ipt)
            if "lora_E" in n:
                ipt_score = self.calculate_score(n, p=p, metric=self.select_metric[1])                
                name_mat = n.replace("lora_E", "%s")
                singular_dict[name_mat] = ipt_score

        all_is = []
        for name_mat in combine_dict: 
            ipt_E = singular_dict[name_mat] 
            ipt_AB = torch.cat(combine_dict[name_mat], dim=1)
            sum_ipt = self._combine_ipt(ipt_E, ipt_AB)
            name_E = name_mat%"lora_E"
            is_dict[name_E] = sum_ipt.view(-1, 1)
            all_is.append(sum_ipt.view(-1))

            # sum_ipt = torch.cat(combine_dict[name_mat], dim=1).sum(dim=1, keepdim=False)
            # name_A = name_mat%"lora_A"
            # name_B = name_mat%"lora_B"
            # hdim_a = self.shape_dict[name_A][1]
            # hdim_b = self.shape_dict[name_B][0]
            # is_dict[name_A] = sum_ipt.view(-1, 1).repeat((1, hdim_a))
            # is_dict[name_B] = sum_ipt.view(1, -1).repeat((hdim_b, 1))
            # all_is.append(sum_ipt.view(-1))

        # torch.use_deterministic_algorithms(False)
        mask_threshold = torch.kthvalue(torch.cat(all_is), (self.total_rank-curr_rank))[0].item()
        # torch.use_deterministic_algorithms(True) 

        with torch.no_grad():
            curr_sum_rank = 0
            sum_param = 0
            for n,p in model.named_parameters():
                if "lora_E" in n: 
                    p.data.masked_fill_(is_dict[n]<=mask_threshold, 0.0)
                    ranknum = (is_dict[n]>mask_threshold).sum().item() 
                    if self.adapt_scaling:
                        model.state_dict()[n.replace("lora_E", "ranknum")].fill_(ranknum)

                    if self.tb_writter is not None and self.global_step % self.log_interval==0:
                        # sum_rank = (p[:, 0] != 0.0).view(-1).sum() 
                        sum_rank = (p.detach() == 0.0).sum().item() 
                        # sum_rank = (is_dict[n][:, 0] > mask_threshold).view(-1).sum() 
                        rec_name = n.split(".")[-5] + "_" + n.split(".")[-4] + "_" + n.split(".")[-2] + "_" + n.split(".")[-1]
                        self.tb_writter.add_scalar("Rank/%s"%(n,), sum_rank, self.global_step) 
                        unmasked_rank = self.lora_init_rank-sum_rank
                        self.tb_writter.add_scalar("Unmask_Rank/%s"%(n,), self.lora_init_rank-sum_rank, self.global_step)
                        self.tb_writter.add_scalar("Ranknum/%s"%(n,), ranknum, self.global_step) 
                        self.rank_pattern[n] = unmasked_rank 
                        curr_sum_rank += unmasked_rank 
                        sum_param += unmasked_rank*self.shape_dict[n.replace("lora_E", "lora_A")][1]  
                        sum_param += unmasked_rank*self.shape_dict[n.replace("lora_E", "lora_B")][0]  

                        # self.tb_writter.add_scalar("Ipt_Sc/Ipt/%s"%rec_name, is_dict[n][:, 0].sum().item(), self.global_step)
                        # self.tb_writter.add_scalar("Ipt_Sc/Sen/%s"%rec_name, self.ipt[n].sum().item(), self.global_step)
                        # self.tb_writter.add_scalar("Ipt_Sc/Unc/%s"%rec_name, self.exp_avg_unc[n].sum().item(), self.global_step)
                        # name_b = n.replace("lora_A", "lora_B")
                        # rec_name_b = rec_name.replace("lora_A", "lora_B")
                        # self.tb_writter.add_scalar("Ipt_Sc/Sen/%s"%rec_name_b, self.ipt[name_b].sum().item(), self.global_step)
                        # self.tb_writter.add_scalar("Ipt_Sc/Unc/%s"%rec_name_b, self.exp_avg_unc[name_b].sum().item(), self.global_step)

            if self.tb_writter is not None and self.global_step%self.log_interval==0:
                self.tb_writter.add_scalar("Prune/total_rank", curr_sum_rank, self.global_step)
                self.tb_writter.add_scalar("Prune/mask_threshold", mask_threshold, self.global_step)
                self.tb_writter.add_scalar("Prune/adarank_sum_param", sum_param, self.global_step)

        return mask_threshold


    def update_and_mask(self, model, global_step):
        if global_step<self.total_step-self.final_warmup or not self.finalize_rank:
            self.update_ipt(model)
        curr_rank, mask_ind = self.schedule_threshold(global_step)
        if mask_ind:
            mask_threshold = self.mask_to_target_rank(model, curr_rank) 
        else:
            mask_threshold = None 
        self._maybe_tb_writter_log(model)

        return curr_rank, mask_threshold

    def _maybe_tb_writter_log(self, model):
        if self.tb_writter is not None and self.global_step%self.log_interval==0:
            if self.plot_cov and self.lora_type == "svd":
                with torch.no_grad():
                    regu_loss = []
                    for n,p in model.named_parameters():
                        if "lora_A" in n or "lora_B" in n:
                            mat = p.data.detach().clone()
                            mat_cov = mat @ mat.T if "lora_A" in n else mat.T @ mat 
                            I = torch.eye(*mat_cov.size(), out=torch.empty_like(mat_cov))
                            I.requires_grad = False
                            orth_regu = torch.norm(mat_cov-I, p="fro")
                            regu_loss.append(orth_regu.item())
                            self.tb_writter.add_scalar("Orth_Regu/%s"%n, orth_regu.item(), self.global_step)
                    self.tb_writter.add_scalar("train/orth_regu_loss", sum(regu_loss)/len(regu_loss), self.global_step)



