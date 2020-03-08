from fastai.vision import *
import os


def run(dataset_path, size, arch, epochs, nm):
    arch = arch.lower()
    path = Path(dataset_path)  # Define path to the image folders
    data = ImageDataBunch.from_folder(path,
                                      valid_pct=0.2,
                                      ds_tfms=get_transforms(),
                                      size=size,
                                      num_workers=6,
                                      bs=64) \
        .normalize(imagenet_stats)
    with open(f'static/encoder/{nm}.plk', 'wb') as f:
        pickle.dump(data.classes, f)
    if arch == 'resnet':
        learn = cnn_learner(data,
                            models.resnet50,
                            metrics=[error_rate, accuracy])
    elif arch == 'alexnet':
        learn = cnn_learner(data,
                            models.alexnet,
                            metrics=[error_rate, accuracy])
    elif arch == 'squeezenet':

        learn = cnn_learner(data,
                            models.squeezenet1_1,
                            metrics=[error_rate, accuracy])
    elif arch == 'densenet':
        learn = cnn_learner(data,
                            models.densenet201,
                            metrics=[error_rate, accuracy])
    elif arch == 'vgg':
        learn = cnn_learner(data,
                            models.vgg19_bn,
                            metrics=[error_rate, accuracy])

    learn.fit_one_cycle(epochs)
    learn.recorder.plot_lr(return_fig=True).savefig(
        f'static/metrics/lr/{nm}.png')
    learn.recorder.plot_losses(return_fig=True).savefig(
        f'static/metrics/loss/{nm}.png')
    learn.recorder.plot_metrics(return_fig=True).savefig(
        f'static/metrics/acc/{nm}.png')
    learn.export(os.path.join(os.getcwd(), 'static', 'models', nm+'.pkl'))
    mp = os.path.join(os.getcwd(), 'static', 'models', nm+'.pkl')
    os.system(f'gsutil cp {mp} gs://model-bucket-dj')

# run('static/datasets/food', 64, 'squeezenet', epochs=2)
