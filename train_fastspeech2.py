import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig, BaseDatasetConfig
from TTS.tts.configs.fastspeech2_config import Fastspeech2Config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.forward_tts import ForwardTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.utils.manage import ModelManager

def formatter(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    txt_file = os.path.join(root_path, meta_file)
    items = []
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0] + ".wav")
            text = cols[1].strip().replace('\u200c', ' ')
            speaker_name = cols[2].strip()
            root_path="/scratch/Akumar/MyTTSDataset_40hrs"
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name,"root_path":root_path})
    return items
output_path = os.path.dirname(os.path.abspath(__file__))

# init configs
dataset_config = BaseDatasetConfig(
    formatter="ljspeech", meta_file_train="metadata_train.csv",meta_file_val="metadata_test.csv" ,path=os.path.join(output_path, "/scratch/Akumar/MyTTSDataset_40hrs/")
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

config = Fastspeech2Config(
    run_name="fastspeech2_ljspeech",
    audio=audio_config,
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=8,
    num_eval_loader_workers=4,
    compute_input_seq_cache=True,
    compute_f0=True,
    f0_cache_path=os.path.join(output_path, "f0_cache"),
    compute_energy=True,
    energy_cache_path=os.path.join(output_path, "energy_cache"),
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="english_cleaners",
    #use_phonemes=True,
    #phoneme_language="en-us",
    #phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    precompute_num_workers=4,
    print_step=50,
    print_eval=False,
    mixed_precision=False,
    max_seq_len=500000,
    output_path=output_path,
    datasets=[dataset_config],
    test_sentences=[
        "మన భూమికి ఉపగ్రహం అయిన చందమామ ఆది కాలంలో గిర్రున తిరిగే భూమి గ్రహం నుంచి విడిపోయిన ఒక పెద్ద శకలం",
        "ఆ తర్వాత నాదల్ మోకాలి గాయం మరింత ఇబ్బంది పెట్టడంతో టోర్నీ నుంచి అర్థాంతరంగా తప్పుకున్నాడు",
        "తన నరాలలో కెనడియన్ ఫ్రెంచ్ తల్లి మరియు అమెరికన్ ఇటాలియన్ తండ్రివైపు రక్తం ప్రవహిస్తుంది",
        "జంతువుల వ్యవసాయం అంతర్గతంగా అసమర్థంగా ఉంటుంది, ఎందుకంటే మొక్కలను ఉత్పత్తి చేసే మొక్కలతో పోల్చితే,  ఉత్పత్తి చేయడానికి ఎక్కువ భూమి, నీరు, ఎరువులు మరియు ఇతరుల వనరులను తీసుకుంటుంది"]
    
)

# compute alignments
if not config.model_args.use_aligner:
    manager = ModelManager()
    model_path, config_path, _ = manager.download_model("tts_models/en/ljspeech/tacotron2-DCA")
    # TODO: make compute_attention python callable
    os.system(
        f"python TTS/bin/compute_attention_masks.py --model_path {model_path} --config_path {config_path} --dataset ljspeech --dataset_metafile metadata.csv --data_path ./recipes/ljspeech/LJSpeech-1.1/  --use_cuda true"
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

# init the model
model = ForwardTTS(config, ap, tokenizer, speaker_manager=None)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)
trainer.fit()
