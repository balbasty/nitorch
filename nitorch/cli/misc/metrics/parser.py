from nitorch.core import cli


help = r"""[nitorch] Compute metrics

usage:
    nitorch metrics -pred *FILES -ref *FILES -m *METHODS [-o *FILES] ...

    -p, --pred              Path to predicted image.
    -r, --ref               Path to reference image.
    -m, --metrics           Metrics to compute. See below.
    -s, --average           Compute mean metrics for multiple labels/images: {'micro', 'macro', 'weighted'}
                            ('weighted' = weighted by reference label volume;
                            label metrics only)
    -x, --exclude [0]       Labels that should never be considered foreground (can be "none")
    -a, --append            Merge results into the output JSON file if it exists
    -o, --output            Output filename (default: {dir}/{base}.metrics.json)
                            Can use {dir}, {sep}, {base} and {ext} of the first
                            predicted image. With several -p/-r pairs, image
                            metrics are keyed by the base name of each
                            predicted image (rather than its index).
        --cpu [N=0]         Use CPU with N threads. If N=0: as many threads as CPUs.
        --gpu [ID=0]        Use GPU
    
metrics for labels
    dice        Dice/F1 score                   = 2 * TP / (2 * TP + FP + FN) 
    jaccard     Jaccard coefficients            = TP / (TP + FP + FN)
    accuracy    Accuracy                        = (TP + TN) / (P + N) 
    ppv         Positive Predictive Value       = TP / (TP + FP)
    npv         Negative Predictive Value       = TN / (TN + FN)
    fdr         False Discovery Rate            = FP / (TP + FP)
    tpr         True Positive Rate              = TP / P
    fpr         False Positive Rate             = FP / N
    fnr         False Negative Rate             = FN / P
    tnr         True Negative Rate              = TN / N
    plr         Positive Likelihood Ratio       = TPR / FPR
    nlr         Negative Likelihood Radio       = FNR / TNR
    hausdorff   Hausdorff distance
    hausdorff95 Hausdorff distance (95th percentile)
    labels      All of the above

metrics for images
    mse         Mean Squared Error
    rmse        Root Mean Squared Error
    mrse        Mean Root Squared Error
    mad         Median Absolute Error           = median(|pred - ref|)
    psnr        Peak Signal-to-Noise Ratio      = 10 * log10(range**2 / MSE)
    ssim        Structural Similarity Index Measure
    images      All of the above

ssim options:
    --ssim-fwhm [3] [vox]   Window size or full-width half max of Gaussian kernel,
                            in voxels (vox) or millimeters (mm).
                            Can be "full" (global statistics).
    --ssim-gauss            Use Gaussian kernel rather than square sliding window
    --range [[MN=0] MX]     Dynamic range (used by psnr and ssim).
                            Default is min/max of reference.
"""

label_metrics = ['labels', 'dice', 'jaccard', 'accuracy', 'ppv', 'npv', 'fdr',
                 'tpr', 'fpr', 'fnr', 'tnr', 'plr', 'nlr', 'hausdorff', 'hausdorff95']
image_metrics = ['images', 'mse', 'rmse', 'mrse', 'mad', 'psnr', 'ssim']
metrics = label_metrics + image_metrics

# Fit command
parser = cli.CommandParser('metrics', help=help)
parser.add_option('pred', ('-p', '--pred'), nargs='+', default=[])
parser.add_option('ref', ('-r', '--ref'), nargs='+', default=[])
parser.add_option('metrics', ('-m', '--metrics'), nargs='+', default=[],
                  validation=cli.Validations.choice(metrics))
parser.add_option('average', ('-s', '--average'), nargs='+', default=[],
                  validation=cli.Validations.choice(['micro', 'macro', 'weighted']))
parser.add_option('exclude', ('-x', '--exclude'), nargs='+', default=[0],
                  convert=cli.Conversions.number_or_str(int))
parser.add_option('append', ('-a', '--append'), nargs='?', default=False,
                  convert=cli.Conversions.bool, action=cli.Actions.store_true)
parser.add_option('output', ('-o', '--output'), nargs=1,
                  default='{dir}{sep}{base}.metrics.json')
parser.add_option('fwhm', '--ssim-fwhm', nargs='+', default=[3],
                  convert=cli.Conversions.number_or_str(float))
parser.add_option('gauss', '--ssim-gauss', nargs='?', default=False,
                  convert=cli.Conversions.bool, action=cli.Actions.store_true)
parser.add_option('range', '--range', nargs='*2', default=[],
                  convert=cli.Conversions.number(float))
parser.add_option('device', '--gpu', nargs='?', default=('cpu', 0),
                  convert=cli.Conversions.device('gpu'),
                  action=cli.Actions.store_value(('gpu', 0)))
parser.add_option('device', '--cpu', nargs='?',
                  convert=cli.Conversions.device('cpu'),
                  action=cli.Actions.store_value(('cpu', 0)))
