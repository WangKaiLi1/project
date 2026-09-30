# 开发日期 ： 2024/5/20
import torch


class Noise(object):
    def __init__(self, args):
        self.args = args
        self.sigma = 0.2
        self.delay = 0.9995
        self.min_sigma = 0.1

        self.target_sigma = 0.1
        self.noise_clip = 0.5
        self.device = args.device

    def GaussianNoise(self, action):
        # 给动作加入均值为0，标准差为sigma的高斯噪声增加探索
        noise = torch.normal(mean=0.0, std=self.sigma, size=action.size()).to(self.device)
        noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
        return action + noise

    def GaussianNoise_target(self, action):
        # 给动作加入均值为0，标准差为sigma的高斯噪声增加探索
        noise = torch.normal(mean=0.0, std=self.target_sigma, size=action.size()).to(self.device)
        noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
        return action + noise
