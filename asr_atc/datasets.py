import torch
import torchaudio
import re
from os import listdir
from os.path import isfile, join
import soundfile as sf
from torch.utils.data.dataset import Dataset
from transformers import Wav2Vec2Processor
from dataclasses import dataclass
from typing import Dict, List, Optional, Union


def remove_special_characters(txt):
    chars_to_ignore_regex = r'[\,\?\.\!\-\;\:\"]'
    return re.sub(chars_to_ignore_regex, '', txt).upper() + " "


def read_transcript(txt_file):
    with open(txt_file, encoding="utf-8", mode="r") as f:
        return remove_special_characters(f.readline().strip())


class ATCDataset(Dataset):

    def __init__(self, path: str, processor: Wav2Vec2Processor, max_length=20):
        self._path = path
        self._processor = processor
        self._wavs = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and f.endswith(".wav")]
        # filter out large wavs...
        filtered = list()
        for wav in self._wavs:
            f = sf.SoundFile(wav)
            l = f.frames / f.samplerate
            if l < max_length:
                filtered.append(wav)
        self._wavs = filtered

    def __len__(self):
        return len(self._wavs)

    def __getitem__(self, index):
        txt = read_transcript(self._wavs[index].replace(".wav", ".txt"))
        speech_array, sampling_rate = sf.read(self._wavs[index])
        speech_array = torch.from_numpy(speech_array).float()
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16_000)
        speech_array = resampler.forward(speech_array).numpy()
        batch = dict()
        batch["input_values"] = self._processor(speech_array, sampling_rate=resampler.new_freq).input_values.squeeze(
            0).tolist()

        with self._processor.as_target_processor():
            batch["labels"] = self._processor(txt).input_ids
        return batch


class CZATCDataset(Dataset):

    def __init__(self, path: str, processor: Wav2Vec2Processor, split: str = "train"):
        self._corpus = []
        self._processor = processor
        with open(f"{join(path, split)}.csv", encoding='utf-8', mode='r') as f:
            for line in list(f)[1:]:
                line = line.strip()
                if not line:
                    continue
                wav, _, txt = line.split(',')
                self._corpus.append((join(path, wav), txt.strip().upper()))

    def __len__(self):
        return len(self._corpus)

    def __getitem__(self, index):
        speech_array, sampling_rate = sf.read(self._corpus[index][0])

        speech_array = torch.from_numpy(speech_array).float()

        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16_000)
        speech_array = resampler.forward(speech_array).numpy()

        batch = dict()
        batch["input_values"] = self._processor(speech_array, sampling_rate=resampler.new_freq).input_values.squeeze(
            0).tolist()

        with self._processor.as_target_processor():
            batch["labels"] = self._processor(self._corpus[index][1]).input_ids
        return batch


# A simple collator for dynamic batching given the long lengths of audio data...
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch
