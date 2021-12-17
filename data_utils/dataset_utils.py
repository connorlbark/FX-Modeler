
from genericpath import exists
from os import listdir, makedirs
from os.path import isfile, isdir, join
import soundfile as sf
from shutil import copyfile, copy

def extract_instrument_name(dirname):
    #translate = {"Gitarre": "guitar", "Bass": "bass"}
    return ''.join(dirname.split(' ')) #remove spaces


def get_instrument_paths(raw_dataset_path, instruments):
    # [insturment] = list of paths to different datasets
    instrument_dirs = {}

    for instrument in instruments:
        fullpath = join(raw_dataset_path, instrument)
        if isdir(fullpath):
            instrument = extract_instrument_name(instrument)

            if instrument not in instrument_dirs:
                instrument_dirs[instrument] = [fullpath]
            else:
                instrument_dirs[instrument].append(fullpath)

    return instrument_dirs


def separate_into_effects(instrument_paths, valid_effects=None):
    # [instrument][effect] = list of sample paths of OVERALL effects; there are multiple effects in each overall effect, so we separate into the individual effects
    effects = {}

    for (instrument, paths) in instrument_paths.items():
        effects[instrument] = {}

        for path in paths:
            effectsdir = join(path, "Samples")
            for effect in listdir(effectsdir):
                # if not a directory, not an effect dir
                if not isdir(join(effectsdir, effect)) or effect == "":
                    continue
    
                if (valid_effects != None and effect not in valid_effects) and (effect != 'NoFX'):
                    continue

                effectdir = join(effectsdir, effect)

                effects[instrument][effect] = {}
                for samplepath in [
                        join(effectdir, p) for p in listdir(effectdir) if isfile(join(effectdir, p))]:
                    file_name = samplepath.split('\\')[-1]

                    effect_num = file_name.split('-')[2]

                    if effect_num in effects[instrument][effect]:
                        effects[instrument][effect][effect_num].append(
                            samplepath)
                    else:
                        effects[instrument][effect][effect_num] = [
                            samplepath]
    return effects


def map_samples_to_output(samples_by_effects, output_dir):
    # list of tuples: (original dataset sample path, directory it should go to, name)
    cleaned_sample_routes = []

    for instrument in samples_by_effects:
        for effect in samples_by_effects[instrument]:
            for effect_num in samples_by_effects[instrument][effect]:
                effect_path = join(output_dir, effect + effect_num)
                for samplepath in samples_by_effects[instrument][effect][effect_num]:
                    name = '-'.join(samplepath.split('\\')[-1].split('-')[0:2])
                    cleaned_sample_routes.append(
                        (samplepath, effect_path, name))

    return cleaned_sample_routes


def compute_sample_mapping(inputdir, outputdir, instruments, effects=None):
    return map_samples_to_output(separate_into_effects(get_instrument_paths(inputdir, instruments), effects), outputdir)

def copy_wav(original_sample_path, output_sample_path):
    # data, samplerate = sf.read(original_sample_path)
    # sf.write(output_sample_path, data, samplerate)
    #print(original_sample_path, output_sample_path)
    copy(original_sample_path, output_sample_path)

def copy_wav_as_chunks(original_sample_path, out_filedir, out_filename, n_samples_in_chunk):
    with sf.SoundFile(original_sample_path) as f:
        for (i, block) in enumerate(f.blocks(blocksize=n_samples_in_chunk, overlap=0, fill_value=0.0)):
            sf.write(join(out_filedir, out_filename+"-"+str(i)+".wav"), block, f.samplerate)

def copy_wav_samples_from_mapping(sample_mapping, n_samples_in_chunk=1024, split_into_chunks=False):
    print('Copying dataset...')
    for (original_sample_path, sample_output_dir, sample_name) in sample_mapping:
        if not exists(sample_output_dir):
            makedirs(sample_output_dir)

        if not split_into_chunks:
            copy_wav(original_sample_path, join(sample_output_dir, sample_name+".wav"))
        else:
            copy_wav_as_chunks(original_sample_path, sample_output_dir, sample_name, n_samples_in_chunk)

