import torch

def configure_scheduler(cfg, optimizer):
    scheduler_name = cfg['training'].get('scheduler', None)
    if scheduler_name is None:
        return None

    if scheduler_name == 'reducelronplateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode=cfg['training']['monitor_mode'], 
            patience=cfg['training']['lr_scheduler_patience'], 
            factor=cfg['training']['lr_scheduler_factor'], 
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': cfg['training']['monitor'],
                'frequency': 1,
                'interval': 'epoch',
            }
        }
    
    elif scheduler_name == 'cosineannealinglr':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            cfg['training']['max_epochs'],
            cfg['training']['min_lr']
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'frequency': 1,
                'interval': 'epoch',
            }
        }
    else:
        return None

def get_optimizer(cfg, parameters):

    optimizer_name = cfg['training']['optimizer'].lower()

    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            parameters,
            lr=cfg['training']['lr'],
            weight_decay=cfg['training'].get('weight_decay', 1e-2)
        )
    else:
        raise ValueError(f'Unsupported optimizer: {optimizer_name}')

    return optimizer