from nitorch.core import cli


help = r"""[nitorch] Distance transform of a mask or label map

usage:
    nitorch distmap *FILES [-l *LABELS] [-t THR] [-i] [-s] [-m METRIC] [-u UNIT] [-o FILE] ...

    -l, --labels *L     Labels that make up the object (default: all voxels above THR)
    -t, --threshold THR Voxels above THR make up the object (default: 0)
    -i, --invert        Swap object and background: measure the distance of
                        background voxels to the object (default: the distance
                        of object voxels to the background)
    -s, --signed        Signed distance: positive inside the object, negative
                        outside it (opposite signs with -i)
    -m, --metric        Distance: {'l2', 'l1'} (default: 'l2')
    -u, --unit          Unit of the distances: {'mm', 'vox'} (default: 'mm')
    -o, --output        Output filenames (default: {dir}{sep}{base}.distmap{ext})
    --cpu [N=0]         Use CPU with N threads. If N=0: as many threads as CPUs.
    --gpu [ID=0]        Use GPU
"""


parser = cli.CommandParser('distmap', help=help)
parser.add_positional('files', nargs='+')
parser.add_option('labels', ('-l', '--labels'), nargs='+', default=[],
                  convert=cli.Conversions.number(float))
parser.add_option('threshold', ('-t', '--threshold'), nargs=1, default=0,
                  convert=cli.Conversions.number(float))
parser.add_option('invert', ('-i', '--invert'), nargs='?', default=False,
                  convert=cli.Conversions.bool, action=cli.Actions.store_true)
parser.add_option('signed', ('-s', '--signed'), nargs='?', default=False,
                  convert=cli.Conversions.bool, action=cli.Actions.store_true)
parser.add_option('metric', ('-m', '--metric'), nargs=1, default='l2',
                  validation=cli.Validations.choice(['l2', 'l1']))
parser.add_option('unit', ('-u', '--unit'), nargs=1, default='mm',
                  validation=cli.Validations.choice(['mm', 'vox']))
parser.add_option('output', ('-o', '--output'), nargs=1,
                  default='{dir}{sep}{base}.distmap{ext}')
parser.add_option('gpu', '--gpu', nargs='?', default=('cpu', 0),
                  convert=cli.Conversions.device('gpu'),
                  action=cli.Actions.store_value(('gpu', 0)))
parser.add_option('gpu', '--cpu', nargs='?',
                  convert=cli.Conversions.device('cpu'),
                  action=cli.Actions.store_value(('cpu', 0)))
