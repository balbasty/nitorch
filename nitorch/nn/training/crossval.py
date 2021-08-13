import torch.cuda
from nitorch.core import py
from .training import ModelTrainer
from torch.multiprocessing import Pool, current_process


def _init1(args):
    trainer, keep_on_gpu, cuda_pool = args
    if cuda_pool:
        procid = (current_process()._identity or [1])[0] - 1
        cuda_id = cuda_pool[procid]
    else:
        cuda_id = None
    if (cuda_id is not None
            and torch.device(trainer.device).type.startswith('cuda')):
        trainer.device = torch.device(f'cuda:{cuda_id}')
    trainer.init()
    if not keep_on_gpu:
        trainer.model = trainer.to(device='cpu')
    return trainer


def _train1(args):
    trainer, epoch, keep_on_gpu, cuda_pool = args
    if cuda_pool:
        procid = (current_process()._identity or [1])[0] - 1
        cuda_id = cuda_pool[procid]
    else:
        cuda_id = None
    if (cuda_id is not None
            and torch.device(trainer.device).type.startswith('cuda')):
        trainer.device = torch.device(f'cuda:{cuda_id}')
    if trainer.initial_epoch < epoch <= trainer.nb_epoch:
        trainer.train1()
        if not keep_on_gpu:
            trainer.model = trainer.to(device='cpu')
    return trainer


def multitrainer(trainers, keep_on_gpu=True, parallel=False, cuda_pool=(0,)):
    """Train multiple models in parallel.

    Parameters
    ----------
    trainers : sequence[ModelTrainer]
    keep_on_gpu : bool, default=True
        Keep all models on GPU (risk of out-of-memory)
    parallel : bool or int, default=False
        Train model in parallel.
        If an int, only this number of models will be trained in parallel.
    cuda_pool : sequence[int], default=[0]
        IDs of GPUs that can be used to dispatch the models.

    """
    n = len(trainers)
    initial_epoch = max(min(trainer.initial_epoch for trainer in trainers), 1)
    nb_epoch = max(trainer.nb_epoch for trainer in trainers)
    trainers = tuple(trainers)
    if parallel:
        parallel = len(trainers) if parallel is True else parallel
        chunksize = max(len(trainers) // (2*parallel), 1)
        pool = Pool(parallel)
        cuda_pool = py.make_list(pool, parallel)

    if not torch.cuda.is_available():
        cuda_pool = []

    if not keep_on_gpu:
        for trainer in trainers:
            trainer.to(device='cpu')

    # --- init ---
    if parallel:
        args = zip(trainers, [keep_on_gpu]*n, [cuda_pool]*n)
        trainers = list(pool.map(_init1, args, chunksize=chunksize))
    else:
        args = zip(trainers, [keep_on_gpu]*n, [cuda_pool]*n)
        trainers = list(map(_init1, args))

    # --- train ---
    for epoch in range(initial_epoch+1, nb_epoch+1):
        if parallel:
            args = zip(trainers, [epoch]*n, [keep_on_gpu]*n, [cuda_pool]*n)
            trainers = list(pool.map(_train1, args, chunksize=chunksize))
        else:
            args = zip(trainers, [epoch]*n, [keep_on_gpu]*n, [cuda_pool]*n)
            trainers = list(map(_train1, args))

