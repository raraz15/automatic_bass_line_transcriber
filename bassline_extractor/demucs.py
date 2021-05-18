import argparse
import sys
from pathlib import Path
import subprocess

import julius
import torch as th
import torchaudio as ta

from .audio import AudioFile, convert_audio_channels
from .pretrained import is_pretrained, load_pretrained
from .utils import apply_model, load_model


def main(wav):

    #parser = argparse.ArgumentParser("demucs.separate",
    #                                 description="Separate the sources for the given tracks")

    #parser.add_argument("tracks", nargs='+', type=Path, default=[], help='Path to tracks')

    #parser.add_argument("-n",
    #                    "--name",
    #                    default="demucs_quantized",
    #                    help="Model name. See README.md for the list of pretrained models. "
    #                         "Default is demucs_quantized.")

    #parser.add_argument("-v", "--verbose", action="store_true")

    #parser.add_argument("-o",
    #                    "--out",
    #                    type=Path,
    #                    default=Path("separated"),
    #                    help="Folder where to put extracted tracks. A subfolder "
    #                    "with the model name will be created.")

    #parser.add_argument("--models",
    #                    type=Path,
    #                    default=Path("models"),
    #                    help="Path to trained models. "
    #                    "Also used to store downloaded pretrained models")
    
    #parser.add_argument("-d",
    #                    "--device",
    #                    default="cuda" if th.cuda.is_available() else "cpu",
    #                    help="Device to use, default is cuda if available else cpu")
    
    
    #parser.add_argument("--shifts",
    #                    default=0,
    #                    type=int,
    #                    help="Number of random shifts for equivariant stabilization."
    #                    "Increase separation time but improves quality for Demucs. 10 was used "
    #                    "in the original paper.")


    #parser.add_argument("--overlap",
    #                    default=0.25,
    #                    type=float,
    #                    help="Overlap between the splits.")

    #parser.add_argument("--no-split",
    #                    action="store_false",
    #                    dest="split",
    #                    default=True,
    #                    help="Doesn't split audio in chunks. This can use large amounts of memory.")
    
    #parser.add_argument("--float32",
    #                    action="store_true",
    #                    help="Convert the output wavefile to use pcm f32 format instead of s16. "
    #                    "This should not make a difference if you just plan on listening to the "
    #                    "audio but might be needed to compute exactly metrics like SDR etc.")
    #parser.add_argument("--int16",
    #                    action="store_false",
    #                    dest="float32",
    #                    help="Opposite of --float32, here for compatibility.")
    #
    
    #parser.add_argument("--mp3", action="store_true",
    #                    help="Convert the output wavs to mp3.")
    #parser.add_argument("--mp3-bitrate",
    #                    default=320,
    #                    type=int,
    #                    help="Bitrate of converted mp3.")

    
    #args = parser.parse_args()


    #name = args.name + ".th"
    name = 'demucs_quantized.th'

# how to download the model?????
    model_path = args.models / name
    if model_path.is_file():
        model = load_model(model_path)
    else:
        if is_pretrained(args.name):
            model = load_pretrained(name)
        else:
            print(f"No pre-trained model {name}", file=sys.stderr)
            sys.exit(1)


    source_names = ["drums", "bass", "other", "vocals"]
    print(f"Separating track")
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()

    sources = apply_model(model, wav, shifts=0, split=True,
                          overlap=0.25, progress=True)
    sources = sources * ref.std() + ref.mean()

    for source, name in zip(sources, source_names):

        if name == 'bass':
            source = source / max(1.01 * source.abs().max(), 1)

            #if args.mp3 or not args.float32:
            #    source = (source * 2**15).clamp_(-2**15, 2**15 - 1).short()

            source = source.cpu()
            #stem = str(track_folder / name)

            return source

        #if args.mp3:
        #    encode_mp3(source, stem + ".mp3",
        #               bitrate=args.mp3_bitrate,
        #               samplerate=model.samplerate,
        #               channels=model.audio_channels,
        #               verbose=args.verbose)
        #else:
        #    wavname = str(track_folder / f"{name}.wav")
        #    ta.save(wavname, source, sample_rate=model.samplerate)