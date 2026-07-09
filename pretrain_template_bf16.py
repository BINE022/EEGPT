
import os
DEBUG_MODE=False
# ========== 环境变量配置 ==========
# 设置环境变量（确保在main函数外）
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["NCCL_NSOCKS_PERTHREAD"] = "4"
os.environ["NCCL_SOCKET_NTHREADS"] = "8"
# 仅在调试时启用阻塞模式
if DEBUG_MODE:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
else:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # 提高训练速度
    
from typing import Any, Dict
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
import pytorch_lightning as pl

from pytorch_lightning import loggers as pl_loggers
import math

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from utils import WarmupCosineSchedule, CosineWDSchedule, grad_logger

from models.modules.utils import NegativeCosineSimilarity, update_momentum, get_collapse_level
# NOTE: ``models.models_liu`` is imported lazily inside the ``CSFM_base`` branch
# below — it is NOT needed for EEGPTV3 / EEGPT_mcae pretraining and was removed
# from module top-level to drop the dependency.

class OrderedCSVLogger(pl.loggers.CSVLogger):
    def log_metrics(self, metrics, step):
        # 固定键的顺序：优先epoch/step，其余按字母排序
        core_keys = ['epoch', 'step']
        other_keys = sorted([k for k in metrics.keys() if k not in core_keys])
        ordered_metrics = {k: metrics[k] for k in core_keys + other_keys if k in metrics}
        
        super().log_metrics(ordered_metrics, step)


####################################################################
# MODEL
class LitEEGPT(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()   
        
        self.cfg = cfg
        if cfg['KEY']=='CSFM_base':
            import models.models_liu as models_liu  # lazy: only needed for CSFM_base
            self.model = models_liu.__dict__['CSFM_base']()
        else:
            
            if ARGS.KEY == 'MAE':
                from models.MAE import MAE, seed_torch, electrode_num
                MODEL_CLASS = MAE
            elif ARGS.KEY == 'EEGPTV2':
                from models.EEGPT_V2 import MAE, CAE, seed_torch, electrode_num
                if ARGS.SKEY == 'CAE':
                    MODEL_CLASS = CAE
                else:
                    MODEL_CLASS = MAE
            elif ARGS.KEY == 'EEGPTV3':
                from models.EEGPT_V3 import MAE, CAE, seed_torch, electrode_num
                if ARGS.SKEY == 'CAE':
                    MODEL_CLASS = CAE
                else:
                    MODEL_CLASS = MAE
            elif ARGS.KEY == 'EEGPT_mcae':
                from models.EEGPT_mcae_opt import MAE, CAE, seed_torch, electrode_num
                if ARGS.SKEY == 'CAE':
                    MODEL_CLASS = CAE
                else:
                    MODEL_CLASS = MAE
                
            self.model =  MODEL_CLASS(
                    max_num_chan  = electrode_num,
                    patch_size    = ARGS.patch_size,
                    output_num    = ARGS.output_num,
                    embed_dim     = ARGS.embed_dim,
                    embed_num     = ARGS.embed_num,
                    depth         = ARGS.depth,
                    num_heads     = ARGS.num_heads,
                    
                    enc_use_all_attn     =ARGS.enc_use_all_attn,
                    enc_use_indp_chan_sum=ARGS.enc_use_indp_chan_sum,
                    enc_use_indp_time_sum=ARGS.enc_use_indp_time_sum,
                    
                    dec_use_all_attn     =ARGS.dec_use_all_attn,
                    dec_use_indp_chan_sum=ARGS.dec_use_indp_chan_sum,
                    dec_use_indp_time_sum=ARGS.dec_use_indp_time_sum,
                    
                    
        
                    predictor_embed_dim     = ARGS.predictor_embed_dim,
                    predictor_embed_dim_inp = ARGS.predictor_embed_dim_inp,
                    predictor_depth         = ARGS.predictor_depth,
                    use_predictor       = ARGS.use_predictor,
                    use_combine_z       = ARGS.use_combine_z,
                )
        if ('ckpt_path' in cfg) and (cfg['ckpt_path'] is not None):
            state_dict = torch.load(cfg['ckpt_path'])['state_dict']
            ckpt = {k[len('model.'):]:v for k,v in state_dict.items()}
            ckpt = {k.replace("_orig_mod.", ''):v for k,v in ckpt.items()}
            self.model.load_state_dict(ckpt)
        if ('use_compile' in cfg) and cfg['use_compile']:
            self.model =  torch.compile(self.model)
        self.loss_fn        = torch.nn.MSELoss()
        self.loss_fn_align  = NegativeCosineSimilarity()
    
    def on_train_start(self):
        # Manually log hyperparameters
        saved_cfg = {k:v for k,v in self.cfg.items() \
            if (not k.startswith('model_')) and (not k.startswith("__"))}
        saved_cfg['test'] = True
        for logger in self.loggers:
            logger.log_hyperparams(saved_cfg)
    def on_validation_start(self):
        # Manually log hyperparameters
        saved_cfg = {k:v for k,v in self.cfg.items() \
            if (not k.startswith('model_')) and (not k.startswith("__"))}
        saved_cfg['test'] = True
        for logger in self.loggers:
            logger.log_hyperparams(saved_cfg)
        return super().on_validation_start()
    def forward_context(self, batch):
        x, chan_ids, x_type = batch
        # -- 
        
        if self.cfg['KEY']=="CSFM_base":
            loss = self.model(x,chan_ids,x_type,0.5,
                        use_avg=ARGS.use_avg_ref).mean()
            return loss,loss,loss
        # -- 
        results = self.model(x, chan_ids, x_type, mNCxy = ARGS.mNCxy, 
                            use_IdentityAE=ARGS.use_IdentityAE,
                            use_avg_ref=ARGS.use_avg_ref,
                            use_time_indp_attn=ARGS.use_time_indp_attn,
                            use_siamese_forward=ARGS.use_siamese_forward
                            )
        if type(results)==tuple: results = [results]
        
        losses = [[], [], [], [], []]
        for result in results:
            target_y_t, target_y_c, r, z_inter = result[:4]
            input_x = result[-1]
            if type(r)==tuple and len(r)==1: r = r[0]
            if ARGS.use_IdentityAE:
                loss = self.loss_fn(input_x, r)
                for x in losses: x.append(loss)
            else:
                loss_a = 0
                collapse = 0
                if ARGS.SKEY=='CAE':
                    pr, pz_m = result[-2]
                    loss_a = self.loss_fn_align(pr, pz_m)
                    collapse = get_collapse_level(pr)
                    
                if type(r)==tuple and len(r)==2:
                    loss_t = self.loss_fn(target_y_t, r[0])
                    loss_c = self.loss_fn(target_y_c, r[1])
                else:
                    loss_t = self.loss_fn(target_y_t, r)
                    loss_c = self.loss_fn(target_y_c, r)
                # print(x.mean(), x.std(), x.min(), x.max())
                # print(target_y_t.mean(), target_y_t.std(), target_y_t.min(), target_y_t.max())
                # print(r.mean(), r.std(), r.min(), r.max())
                
                loss = (loss_t+loss_c)/2 + (ARGS.weight_loss_a * loss_a)
                
                # 实时监控损失值
                if torch.isnan(loss) or torch.isinf(loss):
                    self._handle_nan_loss()
                
                losses[0].append(loss)
                losses[1].append(loss_t)
                losses[2].append(loss_c)
                losses[3].append(loss_a)
                losses[4].append(collapse)
                
        for i in range(len(losses)):
            losses[i] = sum(losses[i])/len(losses[i])
            # losses[i] = losses[i][0]
        return losses
    
    def validation_step(self, batch, batch_idx):
        losses = self.forward_context(batch)
        
        # -- Reconstruct
        self.log('valid_loss' ,   losses[0],   on_epoch=True, on_step=False, sync_dist=True)
        self.log('valid_loss_t' , losses[1], on_epoch=True, on_step=False, sync_dist=True)
        self.log('valid_loss_c' , losses[2], on_epoch=True, on_step=False, sync_dist=True)
        self.log('valid_loss_a' , losses[3], on_epoch=True, on_step=False, sync_dist=True)
        self.log('valid_collapse' , losses[4], on_epoch=True, on_step=False, sync_dist=True)
              
        return losses[0]
    
    def training_step(self, batch, batch_idx):
        losses = self.forward_context(batch)
        
        
        # grad_stats = grad_logger(self.model.named_parameters())
        # self.log('grad_stats.first_layer', grad_stats.first_layer, on_epoch=True, on_step=False, sync_dist=True)
        # self.log('grad_stats.last_layer',  grad_stats.last_layer, on_epoch=True, on_step=False, sync_dist=True)
        # if grad_stats.max>float('-inf'):
        #     self.log('grad_stats.max', grad_stats.max, on_epoch=True, on_step=False, sync_dist=True)

        # if grad_stats.min<float('inf'): 
        #     self.log('grad_stats.min', grad_stats.min, on_epoch=True, on_step=False, sync_dist=True)
        # # print(grad_stats.max)
        # #############################################################################
        # # 记录数据分布
        # if batch_idx==1:
        #     tensorboard = self.loggers[0].experiment
        #     for name, param in self.model.named_parameters():  # 返回网络的
        #         if isinstance(param, nn.Parameter) and len(param.shape)>0: 
        #             tensorboard.add_histogram(name + '_data', param, self.current_epoch)
        #             if param.requires_grad and (not (param.grad is None)):
        #                 tensorboard.add_histogram(name + '_grad', param.grad, self.current_epoch)
        # #############################################################################
        
        # -- Reconstruct
        self.log('train_loss' ,   losses[0],   on_epoch=True, on_step=False, sync_dist=True)
        self.log('train_loss_t' , losses[1], on_epoch=True, on_step=False, sync_dist=True)
        self.log('train_loss_c' , losses[2], on_epoch=True, on_step=False, sync_dist=True)
        self.log('train_loss_a' , losses[3], on_epoch=True, on_step=False, sync_dist=True)
        self.log('train_collapse',losses[4], on_epoch=True, on_step=False, sync_dist=True)
        return losses[0]
    
    def _handle_nan_loss(self):
        """NaN损失处理策略"""
        print("检测到NaN/Inf损失! 保存当前状态...")
        torch.save({
            'model': self.state_dict(),
            # 'optimizer': self.optimizers().state_dict()
        }, "nan_debug_checkpoint.pt")
        
        # 可选：停止训练或调整学习率
        # self.trainer.should_stop = True
        
    def on_train_batch_start(self, batch: Any, batch_idx: int):
        self.wd_scheduler.step()
        return super().on_train_batch_start(batch, batch_idx)
    
    
    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        # momentum update of momentum encoder and projection head
        if ARGS.SKEY=='CAE':
            m = next(self.momentum_scheduler)
            update_momentum(self.model.model.encoder, self.model.encoder_momentum, m)
            update_momentum(self.model.projection_head, self.model.projection_head_momentum, m)
        
        return super().on_train_batch_end(outputs, batch, batch_idx)
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        res = super().on_load_checkpoint(checkpoint)
        self.configure_optimizers()
        return res
    
    def configure_optimizers(self):
        no_names = self.model.no_weight_decay()
        def is_in(x):
            for a in no_names:
                if a in x:return True
            return False
        param_groups = [
            {
                'params': (p for n, p in self.model.named_parameters()
                        if (('bias' not in n) and (len(p.shape) != 1) and (not is_in(n))) and ('chan_embed' not in n) and ('momentum' not in n))
            }, {
                'params': (p for n, p in self.model.named_parameters()
                        if (('bias' in n) or (len(p.shape) == 1) or is_in(n)) and ('chan_embed' not in n) and ('momentum' not in n)),
                'WD_exclude': True,
                'weight_decay': 0
            }, {
                'params': (p for n, p in self.model.chan_embed.named_parameters()),
                'WD_exclude': True,
                'weight_decay': 0
            },
        ]
        
        optimizer = torch.optim.AdamW(param_groups, lr=6e-5)        
        
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=ARGS.max_lr, steps_per_epoch=ARGS.steps_per_epoch, 
                                                           epochs=ARGS.max_epochs,
                                                           div_factor = 2,
                                                           final_div_factor=8,
                                                           pct_start = 0.2 ,
                                                           )
        # lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.)
        lr_dict = {
            'scheduler': lr_scheduler, # The LR scheduler instance (required)
            # The unit of the scheduler's step size, could also be 'step'
            'interval': 'step',
            'frequency': 1, # The frequency of the scheduler
            'monitor': 'valid_loss', # Metric for `ReduceLROnPlateau` to monitor
            'strict': True, # Whether to crash the training if `monitor` is not found
            'name': None, # Custom name for `LearningRateMonitor` to use
        }
        self.wd_scheduler = CosineWDSchedule(
                            optimizer,
                            ref_wd=1e-6,
                            final_wd=1e-6,
                            T_max=int(ARGS.max_epochs*ARGS.steps_per_epoch))
        
        if ARGS.SKEY=='CAE':
            ema = [0.996,1.0]
            self.momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ARGS.steps_per_epoch*ARGS.max_epochs)
                            for i in range(int(ARGS.steps_per_epoch*ARGS.max_epochs)+1))
        return (
            {'optimizer': optimizer, 'lr_scheduler': lr_dict},
        )


####################################################################
# ARGS

class ARGS:
    # ckpt_path = "./logs_EEGPTV2/checkpoints/best-EEGPTV2_v2_IAE-epoch=199-valid_loss=0.8867.ckpt"
    # ckpt_path = "logs_EEGPTV2/checkpoints/last-EEGPTV2_v2_IAE-epoch=199-valid_loss=0.6394.ckpt"
    # ckpt_path = "logs_EEGPTV2/checkpoints/last-EEGPTV2_v2_IAE-epoch=199-valid_loss=0.6374.ckpt"
    
    # -- 
    # ckpt_path = "logs_EEGPTV2/IAE_embed_num_2_2_2/checkpoints/best-EEGPTV2_v2_IAE-epoch=189-valid_loss=0.0575.ckpt"
    # ckpt_path="logs_MAE/checkpoints/last-MAE_v103_v0_MAE-epoch=31-valid_loss=0.1162.ckpt"
    # -- 
    seed = 2024
    version = None
    # KEY = 'MAE'
    # KEY = 'CSFM_base'
    # KEY = 'EEGPTV2'
    KEY = 'EEGPT_mcae'
    SKEY= 'CAE'
    
    max_epochs= 32
    steps_per_epoch = None
    max_lr    = 1e-3
    batch_size= 32*3*2
    early_stop= 5
    log_path  = None
    log_name  = None
    save_path = None
    save_name = None
    use_IdentityAE = False
    devices   = [0,1,2]
    
    weight_loss_a = 1.0
    
    # -- dataset
    percentile_q=-1#0.95
    use_tanh    =False
    num_workers = 20
    use_tiny_data=False
    use_fake_data=False
    # -- model
    patch_size    = 50
    output_num    = 2 
    embed_dim     = 256
    embed_num     = (2,2,2)
    depth         = 8
    num_heads     = 16
    # mNCxy         = (5,5,5,5) # [mN_x, mC_x, mN_y, mC_y]
    mNCxy         = (10,13,10,13)
    use_avg_ref   = True
    
    
    enc_use_all_attn     =False
    enc_use_indp_chan_sum=True
    enc_use_indp_time_sum=True
    
    dec_use_all_attn     =False
    dec_use_indp_chan_sum=True
    dec_use_indp_time_sum=True
    
    use_time_indp_attn   =True
    use_siamese_forward  =True
    # -- phase
    train_phase = False
    valid_phase = True
    use_compile = False
    
##################################################################################
def init():
    ARGS.use_IdentityAE=(ARGS.SKEY == 'IAE')

    percentile_q = str(1 if ARGS.percentile_q>0 else 0)
    use_tanh     = str(1 if ARGS.use_tanh else 0)
    use_tiny_data     = str(1 if ARGS.use_tiny_data else 0)
    use_fake_data     = str(1 if ARGS.use_fake_data else 0)
    use_avg_ref     = str(1 if ARGS.use_avg_ref else 0)
    use_compile     = str(1 if ARGS.use_compile else 0)
    use_IdentityAE  = str(1 if ARGS.use_IdentityAE else 0)
    use_siamese_forward = str(1 if ARGS.use_siamese_forward else 0)
    
    
    enc_use_all_attn     = str(1 if ARGS.enc_use_all_attn else 0)
    enc_use_indp_chan_sum     = str(1 if ARGS.enc_use_indp_chan_sum else 0)
    enc_use_indp_time_sum  = str(1 if ARGS.enc_use_indp_time_sum else 0)
    
    dec_use_all_attn     = str(1 if ARGS.dec_use_all_attn else 0)
    dec_use_indp_chan_sum     = str(1 if ARGS.dec_use_indp_chan_sum else 0)
    dec_use_indp_time_sum  = str(1 if ARGS.dec_use_indp_time_sum else 0)


    ARGS.version   = f"d{str(ARGS.depth)}_e{str(ARGS.embed_dim)}_{''.join([str(x) for x in ARGS.embed_num])}_\
A{enc_use_all_attn}{enc_use_indp_chan_sum}{enc_use_indp_time_sum}{dec_use_all_attn}{dec_use_indp_chan_sum}{dec_use_indp_time_sum}_\
q{percentile_q}th{use_tanh}t{use_tiny_data}f{use_fake_data}a{use_avg_ref}c{use_compile}i{use_IdentityAE}o{ARGS.output_num}siam{use_siamese_forward}s{str(ARGS.seed)}"
    ARGS.log_path  = f'./logs_{ARGS.KEY}/'
    ARGS.log_name  = f'{ARGS.KEY}_{ARGS.SKEY}'
    ARGS.save_path = f'./logs_{ARGS.KEY}/checkpoints'
    ARGS.save_name = f'{ARGS.KEY}_{ARGS.SKEY}_{ARGS.version}'+'-ep{epoch}-vs{valid_loss:.4f}'

        
    from models.EEGPT_V2 import seed_torch
    seed_torch(ARGS.seed)

def main():
    ####################################################################
    # Dataset
    from dataset.dataset_pretrain import get_data, DataLoader

    train_dataset,valid_dataset,test_dataset = get_data(percentile_q=ARGS.percentile_q, 
                                                        use_tanh=ARGS.use_tanh, 
                                                        use_tiny_data=ARGS.use_tiny_data,
                                                        use_fake_data=ARGS.use_fake_data)

    train_loader = DataLoader(train_dataset,num_workers=ARGS.num_workers,  pin_memory=True, batch_size = ARGS.batch_size,shuffle = True,  drop_last=True)
    # valid_loader = DataLoader(valid_dataset,num_workers=ARGS.num_workers,  batch_size = ARGS.batch_size,  shuffle = False, drop_last=False)
    valid_loader = DataLoader(test_dataset, num_workers=ARGS.num_workers,  batch_size = ARGS.batch_size,  shuffle = False, drop_last=False)

    ARGS.steps_per_epoch = math.ceil(len(train_loader)/len(ARGS.devices))

    ##################################################################

    # init model
    torch.set_float32_matmul_precision("medium")
    model = LitEEGPT(ARGS.__dict__)

    ckpt_callback = ModelCheckpoint(
        monitor='valid_loss',
        save_top_k=1,
        mode='min',
        dirpath=ARGS.save_path,
        filename='best-{}'.format(ARGS.save_name),
        auto_insert_metric_name=False
    )

    last_callback = ModelCheckpoint(
        every_n_epochs=ARGS.max_epochs,
        save_top_k=1,
        dirpath=ARGS.save_path,
        filename='last-{}'.format(ARGS.save_name),
        auto_insert_metric_name=False
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    callbacks = [lr_monitor, ckpt_callback, last_callback]

    if ARGS.train_phase:
        trainer = pl.Trainer(   
                                gradient_clip_val = 1.0,
                                gradient_clip_algorithm="norm",
                                strategy='auto',
                                #  strategy='ddp_find_unused_parameters_true', 
                                #precision="bf16-mixed",
                                precision="bf16-mixed",  # 优先使用BFloat16
                                # precision=32, 
                                # detect_anomaly=True,  # 启用PyTorch异常检测
                                devices=ARGS.devices, 
                                max_epochs=ARGS.max_epochs, 
                                callbacks=callbacks,
                                logger=[
                                    pl_loggers.TensorBoardLogger(ARGS.log_path, name=ARGS.log_name+"_tb", version=ARGS.version), 
                                    pl_loggers.CSVLogger(ARGS.log_path, name=ARGS.log_name+"_csv", version=ARGS.version)])
        trainer.fit(model, train_loader, valid_loader, )

    if ARGS.valid_phase:
        
        trainer = pl.Trainer(   
                                gradient_clip_val = 1.0,
                                gradient_clip_algorithm="norm",
                                strategy='auto',
                                #  strategy='ddp_find_unused_parameters_true', 
                                precision="bf16-mixed",  # 优先使用BFloat16
                                # precision=32, 
                                detect_anomaly=True,  # 启用PyTorch异常检测
                                devices=ARGS.devices, 
                                max_epochs=ARGS.max_epochs, 
                                # callbacks=callbacks,
                                logger=[
                                    # pl_loggers.CSVLogger(ARGS.log_path, name=ARGS.log_name+"_valid_csv")
                                    OrderedCSVLogger(ARGS.log_path, name=ARGS.log_name+"_valid_csv")
                                    ])
        trainer.validate(model, valid_loader)
        # trainer.validate(model, train_loader)
        
    # CUDA_VISIBLE_DEVICES=2,3,5 python pretrain_template.py