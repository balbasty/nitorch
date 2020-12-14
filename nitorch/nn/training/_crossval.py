from ._training import ModelTrainer


def multitrainer(trainers, keep_on_gpu=False):
    """Train multiple models in parallel.

    Parameters
    ----------
    trainers : sequence[ModelTrainer]
    keep_on_gpu : bool, default=False
        Keep all models on GPU (risk of out-of-memory)

    """

    initial_epoch = max(min(trainer.initial_epoch for trainer in trainers), 1)
    nb_epoch = max(trainer.nb_epoch for trainer in trainers)

    if not keep_on_gpu:
        for trainer in trainers:
            trainer.model = trainer.model.cpu()

    for trainer in trainers:
        trainer.init()
        if not keep_on_gpu:
            trainer.model = trainer.model.cpu()

    for epoch in range(initial_epoch, nb_epoch+1):
        for trainer in trainers:
            trainer.train1()
            if not keep_on_gpu:
                trainer.model = trainer.model.cpu()

