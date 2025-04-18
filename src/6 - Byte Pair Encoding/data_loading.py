import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from pathlib import Path
from midi_encoding import *
import torch

class MidiDataset(Dataset):
    def __init__(self, file_names, midi_path, score_path, sample_length, max_file_length):
        self.file_names = file_names
        self.data = None
        self.file_lengths = None
        self.total_samples = 0
        self.sample_length = sample_length
        self.midi_path = midi_path
        self.score_path = score_path
        self.max_file_length = max_file_length

    @torch.no_grad()
    def load_samples(self, vocab, device):
        data = []
        file_lengths = []
        for file_name in self.file_names:

            midi_file_path = Path(self.midi_path, file_name)
            score_file_path = Path(self.score_path, file_name)
            encoded_file_path = Path(self.score_path, f'{file_name}.npy')

            if (encoded_file_path.exists()):
                # print(f'Loading {score_file_path}')
                idx_score = np.load(encoded_file_path, allow_pickle=True)
            else:
                # print(f'Processing {midi_file_path}')
                idx_score = midifile_to_idx_score(midi_file_path, vocab)
                if (idx_score is None): # Skip files that could not be processed
                    # print(f'Could not process {midi_file_path}')
                    continue
                np.save(score_file_path, idx_score)

            samples = []
            
            # Split idx_score into blocks of size sample_length, padding the last blocks if necessary
            for i in range(0, len(idx_score), self.sample_length):
                block = idx_score[i:i + self.sample_length]
                if len(block) < self.sample_length:
                    last_tidx = block[-1, 1]
                    pad_tidx = last_tidx + 1
                    padding_count = self.sample_length - len(block)
                    padding = np.stack([[vocab.pad_idx, pad_tidx]] * padding_count)
                    block = np.concatenate([block, padding])

                samples.append(block)

            # Skip files that are empty or too long
            if len(samples) == 0 or len(samples) > self.max_file_length:
                continue
            
            data.append(torch.tensor(np.array(samples), device=device))
            file_lengths.append(len(samples))
        
        self.total_samples = sum(file_lengths)
        self.data = torch.nested.nested_tensor(data, device=device)
        self.file_lengths = torch.tensor(file_lengths, device=device)
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        file_idx = idx[0]
        sample_idx = idx[1]
        sample = self.data[file_idx, sample_idx] # = self.data[idx]
        return file_idx, sample