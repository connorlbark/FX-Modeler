from genericpath import exists
from os import listdir
from os.path import isfile, isdir, join
import argparse
import dataset_utils

# --------------------------- ARGUMENTS ---------------------------
parser = argparse.ArgumentParser(description='Generate dataset.')
parser.add_argument('inpath', help='path to directory of IDMT dataset.')
parser.add_argument('outpath', help='path to directory of output.')

parser.add_argument('--instruments', default='all',
                    help='Specify the instruments to use in a comma separated list. Defaults to \'all\'')
parser.add_argument('--effects', default='all',
                    help='Specify the effects to use in a comma separated list. Defaults to \'all\'')

parser.add_argument('--chunk', action='store_true',
                    help='Copies the sound file directly; does not chunk into uniformly sized wav files.')
parser.add_argument('--n-samples', default=1024, type=int,
                    help='The size of the sound file if chunking.')

args = parser.parse_args()

# check if instruments given are valid (i.e., they actually exist.)
valid_instruments = [d for d in listdir(
    args.inpath) if isdir(join(args.inpath, d))]
instruments = []
if args.instruments == 'all':
    instruments = valid_instruments
else:
    instruments = args.instruments.split(',')
    for inst in instruments:
        if inst not in valid_instruments:
            print('Invalid instrument:', inst)
            print('Valid instruments:', valid_instruments)
            exit(1)

effects = None if args.effects == 'all' else args.effects.split(',') #todo: actually check if these exist

# --------------------------- COPY SAMPLES ---------------------------

dataset_utils.copy_wav_samples_from_mapping(
    dataset_utils.compute_sample_mapping(args.inpath, args.outpath, instruments, effects=effects),
    n_samples_in_chunk=args.n_samples, split_into_chunks=args.chunk)
