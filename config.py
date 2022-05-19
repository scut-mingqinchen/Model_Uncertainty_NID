import argparse
from utils.common_utils import makedirs


class Config:
    def __init__(self, model_name, is_gaussian=False, sigma=0.00, is_poisson=False, peak=1024., is_noise_blind=False,
                 gamma=1., nn_last_act='Sigmoid', n_iteration=100000, degradtion_type='conv'):
        self.parser = argparse.ArgumentParser(description='IJCV2022_MUNID')
        self.parser.add_argument('--model_name', type=str, default=model_name)
        self.parser.add_argument('--is_gaussian', type=bool, default=is_gaussian)
        self.parser.add_argument('--sigma', type=float, default=sigma)
        self.parser.add_argument('--is_poisson', type=bool, default=is_poisson)
        self.parser.add_argument('--peak', type=float, default=peak)
        self.parser.add_argument('--is_noise_blind', type=bool, default=is_noise_blind)
        self.parser.add_argument('--gamma', type=float, default=gamma)
        self.parser.add_argument('--nn_last_act', type=str, default=nn_last_act)
        self.parser.add_argument('--n_iteration', type=int, default=n_iteration)
        self.parser.add_argument('--degradtion_type', type=str, default=degradtion_type)
        self.parser.add_argument('--lr', type=float, default=1e-4)
        self.parser.add_argument('--n_save_freq', type=int, default=1000)
        self.parser.add_argument('--p_m', type=float, default=0.30)
        self.parser.add_argument('--eta_stabilizer', type=float, default=5e-4)
        self.parser.add_argument('--n_inference', type=int, default=50)

        self.model_dir = './results/' + model_name
        self.model_save_dir = self.model_dir + '/model/'
        self.parser.parse_args(namespace=self)
        makedirs([self.model_dir, self.model_save_dir])
