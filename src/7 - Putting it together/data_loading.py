import numpy as np
from torch.utils.data import Dataset, Sampler
from pathlib import Path
from midi_encoding import *
import torch
from itertools import cycle
from multiprocessing import Pool
from functools import partial

def encode_file(vocab, score_path, midi_file_path):
    try:
        file_name = midi_file_path.name
        score_file_path = Path(score_path, file_name)
        encoded_file_path = Path(score_path, f'{file_name}.npy')
        if not encoded_file_path.exists():
            idx_score = midifile_to_idx_score(midi_file_path, vocab)
            if (idx_score is not None):
                np.save(score_file_path, idx_score)
    except Exception as e:
        print(f'Error: {e}')

class MidiDataset(Dataset):
    def __init__(self, vocab, midi_file_paths, score_path, sample_length, max_file_length):
        self.midi_file_paths = midi_file_paths
        self.data = None
        self.file_lengths = None
        self.total_samples = 0
        self.sample_length = sample_length
        self.score_path = score_path
        self.max_file_length = max_file_length
        self.vocab = vocab

    def ensure_encoded(self):
        partial_encode_file = partial(encode_file, self.vocab, self.score_path)
        if __name__ == '__main__':
            with Pool(processes=32) as pool:  # Adjust the number of processes based on your system
                pool.map(partial_encode_file, self.midi_file_paths)

    @torch.no_grad()
    def load_samples(self, device):
        self.ensure_encoded()
        data = []
        file_lengths = []
        for midi_file_path in self.midi_file_paths:
            file_name = midi_file_path.name
            encoded_file_path = Path(self.score_path, f'{file_name}.npy')
            idx_score = np.load(encoded_file_path, allow_pickle=True)

            samples = []
            
            # Split idx_score into blocks of size sample_length, padding the last blocks if necessary
            for i in range(0, len(idx_score), self.sample_length):
                block = idx_score[i:i + self.sample_length]
                if len(block) < self.sample_length:
                    last_tidx = block[-1, 1]
                    pad_tidx = last_tidx + 1
                    padding_count = self.sample_length - len(block)
                    padding = np.stack([[self.vocab.pad_idx, pad_tidx]] * padding_count)
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
    
class ContiguousBatchSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.batches = []
    
    def precompute_indices(self, batch_size):
        
        file_count = len(self.dataset.file_lengths)
        if file_count < batch_size:
            raise ValueError('The number of files must be greater than or equal to the batch size, as files must be spread across a single batch dimension.')
        
        file_idxs = list(range(batch_size))
        file_positions = [0] * batch_size

        while True:
            batch = []
            for batch_idx in range(batch_size):
                
                current_file_idx = file_idxs[batch_idx]
                current_file_position = file_positions[batch_idx]
                current_file_length = self.dataset.file_lengths[current_file_idx]
                
                # Check if the current file is exhausted
                if current_file_position == current_file_length:
                    # Find the next file that hasn't been started
                    files_exhausted = True
                    min_file_index = max(file_idxs) + 1
                    for next_file_idx in range(min_file_index, file_count):
                        if self.dataset.file_lengths[next_file_idx] > 0:
                            current_file_idx = next_file_idx
                            current_file_position = 0
                            file_idxs[batch_idx] = current_file_idx
                            file_positions[batch_idx] = current_file_position
                            files_exhausted = False
                            break
                    
                    if files_exhausted:
                        return

                batch.append([current_file_idx, current_file_position])                
                file_positions[batch_idx] += 1

            self.batches.append(batch)

    def __iter__(self):
        for batch in cycle(self.batches):
            yield batch

    def __len__(self):
        return len(self.batches)
    
