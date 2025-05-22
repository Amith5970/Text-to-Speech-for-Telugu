import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# from TTS.tts.datasets.tokenizer import Tokenizer

def formatter(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    txt_file = os.path.join(root_path, meta_file)
    items = []
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0] + ".wav")
            text = cols[1].strip().replace('\u200c', ' ')
            speaker_name = cols[2].strip()
            root_path="/scratch/Akumar/TTS/MyTTSDataset"
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name,"root_path":root_path})
    return items
output_path = os.path.dirname(os.path.abspath(__file__))

# init configs
dataset_config = BaseDatasetConfig(
    formatter="ljspeech", meta_file_train="metadata_train.csv",meta_file_val="metadata_test.csv" ,path=os.path.join(output_path, "/scratch/Akumar/TTS/MyTTSDataset/")
)

audio_config = BaseAudioConfig(
    sample_rate=22050,
    do_trim_silence=True,
    trim_db=60.0,
    signal_norm=False,
    mel_fmin=0.0,
    mel_fmax=8000,
    spec_gain=1.0,
    log_func="np.log",
    ref_level_db=20,
    preemphasis=0.0,
)

config = Tacotron2Config(  # This is the config that is saved for the future use
    audio=audio_config,
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    ga_alpha=0.0,
    decoder_loss_alpha=0.25,
    postnet_loss_alpha=0.25,
    postnet_diff_spec_alpha=0,
    decoder_diff_spec_alpha=0,
    decoder_ssim_alpha=0,
    postnet_ssim_alpha=0,
    r=2,
    attention_type="dynamic_convolution",
    double_decoder_consistency=False,
    epochs=2500,
    text_cleaner="no_cleaners",
    use_phonemes=False,
    phoneme_language="en-us",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    print_step=25,
	save_step=10000,
    print_eval=True,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
	  test_sentences=[
        ["మన భూమికి ఉపగ్రహం అయిన చందమామ ఆది కాలంలో గిర్రున తిరిగే భూమి గ్రహం నుంచి విడిపోయిన ఒక పెద్ద శకలం"],
        ["ఆ తర్వాత నాదల్ మోకాలి గాయం మరింత ఇబ్బంది పెట్టడంతో టోర్నీ నుంచి అర్థాంతరంగా తప్పుకున్నాడు"],
        ["తన నరాలలో కెనడియన్ ఫ్రెంచ్ తల్లి మరియు అమెరికన్ ఇటాలియన్ తండ్రివైపు రక్తం ప్రవహిస్తుంది"],
        ["జంతువుల వ్యవసాయం అంతర్గతంగా అసమర్థంగా ఉంటుంది, ఎందుకంటే మొక్కలను ఉత్పత్తి చేసే మొక్కలతో పోల్చితే,  ఉత్పత్తి చేయడానికి ఎక్కువ భూమి, నీరు, ఎరువులు మరియు ఇతరుల వనరులను తీసుకుంటుంది"]
    ],
)

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# If characters are not defined in the config, default characters are passed to the config
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
	formatter=formatter,
)

# INITIALIZE THE MODEL
# Models take a config object and a speaker manager as input
# Config defines the details of the model like the number of layers, the size of the embedding, etc.
# Speaker manager is used by multi-speaker models.
model = Tacotron2(config, ap, tokenizer)

# INITIALIZE THE TRAINER
# Trainer provides a generic API to train all the 🐸TTS models with all its perks like mixed-precision training,
# distributed training, etc.
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)

# AND... 3,2,1... 🚀
trainer.fit()
