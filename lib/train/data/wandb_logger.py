from collections import OrderedDict

try:
    import wandb
except ImportError:
    raise ImportError(
        'Please run "pip install wandb" to install wandb')


class WandbWriter:
    def __init__(self, exp_name, cfg, output_dir, cur_step=0, step_interval=0):
        self.wandb = wandb
        self.step = cur_step
        self.interval = step_interval
        
        # wandb.init(project="tracking", name=exp_name, config=cfg, dir=output_dir) # 有时候初始化会出错，这里可以多初始化几次
        for i in range(10):
            try:
                wandb.init(project="tracking", name=exp_name, config=cfg, dir=output_dir)
                break
            except:
                print('wandb init failed')
                continue

    def write_log(self, stats: OrderedDict, epoch=-1):
        self.step += 1
        for loader_name, loader_stats in stats.items():
            if loader_stats is None:
                continue

            log_dict = {}
            for var_name, val in loader_stats.items():
                if hasattr(val, 'avg'):
                    log_dict.update({loader_name + '/' + var_name: val.avg})
                else:
                    log_dict.update({loader_name + '/' + var_name: val.val})

                if epoch >= 0:
                    log_dict.update({loader_name + '/epoch': epoch})

            self.wandb.log(log_dict, step=self.step*self.interval)
