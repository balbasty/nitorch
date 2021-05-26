import torch.cuda

from .training import ModelTrainer
from multiprocessing import Process


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

    initial_epoch = max(min(trainer.initial_epoch for trainer in trainers), 1)
    nb_epoch = max(trainer.nb_epoch for trainer in trainers)
    trainers = tuple(trainers)
    if parallel:
        parallel = len(trainers) if parallel is True else parallel
        chunks = (len(trainers)+1) // parallel

    if not torch.cuda.is_available():
        cuda_pool = []

    if not keep_on_gpu:
        for trainer in trainers:
            trainer.to(device='cpu')

    for trainer in trainers:
        trainer.init()
        if not keep_on_gpu:
            trainer.model = trainer.to(device='cpu')

    for epoch in range(initial_epoch+1, nb_epoch+1):

        def train1(args):
            trainer, cuda_id = args
            if (cuda_id is not None
                    and torch.device(trainer.device).type.startswith('cuda')):
                trainer.device = torch.device(f'cuda:{cuda_id}')
            if trainer.initial_epoch < epoch <= trainer.nb_epoch:
                trainer.train1()
                if not keep_on_gpu:
                    trainer.model = trainer.to(device='cpu')

        if parallel:
            for chunk in range(chunks):
                subtrainers = trainers[chunk*parallel:(chunk+1)*parallel]
                if cuda_pool:
                    cuda_ids = [cuda_pool[i % len(cuda_pool)]
                                for i in range(len(subtrainers))]
                else:
                    cuda_ids = [None] * len(subtrainers)
                p = Process(target=train1, args=zip(subtrainers, cuda_ids))
                p.start()
                p.join()
        else:
            for trainer in trainers:
                train1((trainer, None))

