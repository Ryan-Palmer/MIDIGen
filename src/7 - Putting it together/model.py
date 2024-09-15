import torch
import torch.nn.functional as F
from pathlib import Path
import music21 as m21
musescore_path = '/usr/bin/mscore'
m21.environment.set('musicxmlPath', musescore_path)
m21.environment.set('musescoreDirectPNGPath', musescore_path)
from midi_encoding import *
from einops import rearrange, repeat, pack, unpack, einsum
import faiss
import math
import faiss.contrib.torch_utils

cache_size_gb = 8
resources = faiss.StandardGpuResources() 
resources.setTempMemory(cache_size_gb * 1024 * 1024 * 1024)

class KNN():

    @torch.no_grad()
    def __init__(self, dim, max_memories, device):
        self.dim = dim
        self.max_memories = max_memories
        self.shape = (max_memories, 2, dim)
        self.db_offset = 0
        self.db = torch.zeros(self.shape, dtype = torch.float32, device=device)
        self.index = faiss.GpuIndexFlatL2(resources, dim)
        self.device = device

    @torch.no_grad()
    def add_to_db(self, new_data):
        new_data_len = new_data.shape[0] # (t)

        if new_data_len > self.max_memories:
            raise ValueError('Batch size exceeds memory limit.')

        ids = torch.arange(new_data_len) + self.db_offset

        self.db[ids] = new_data.detach()
        self.db_offset += new_data_len

    @torch.no_grad()
    def add(self, new_data):
        self.add_to_db(new_data)
        keys, vals = new_data.unbind(dim=-2) # Only keys are used in knn index
        self.index.add(keys.detach().contiguous()) # (t, c)

    @torch.no_grad()
    def search(self, query, top_k):   
             
        T, C = query.shape

        # If we have enough memories, search and retrieve, otherwise return zeros",
        if self.index.ntotal >= top_k:
            # The tooltip says the args are (n, x, k) but that's the CPP api, it's actually (x, k) in Python (n is the first dim of x anyway so can be inferred).
            distances, indices = self.index.search(query.detach(), top_k)
            kvs = self.db[indices]
        else:
            kvs = torch.zeros((T, top_k, 2, C), device=self.device)

        return kvs

    @torch.no_grad()
    def clear(self):
        self.index.reset()
        self.db = torch.zeros(self.shape, dtype = torch.float32, device=self.device)
        self.db_offset = 0

class XLRelativePosition(torch.nn.Module):
  def __init__(
      self,
      n_buckets,
      max_distance,
      n_head,
      scaling_factor,
      device):
    
    super().__init__()
    self.scale = scaling_factor
    self.num_buckets = n_buckets
    self.max_distance = max_distance
    self.device = device
    self.relative_attention_embedding = torch.nn.Embedding(n_buckets, n_head)

  def relative_position_bucket(self, relative_position_matrix):
    inv_rel_pos = -relative_position_matrix
    masked_rel_pos = torch.max(inv_rel_pos, torch.zeros_like(inv_rel_pos, device=self.device))

    max_exact = self.num_buckets // 2

    is_small = masked_rel_pos < max_exact
    val_if_large = max_exact + (torch.log(masked_rel_pos.float() / max_exact) / math.log(self.max_distance / max_exact) * (self.num_buckets - max_exact)).long()

    # Clip the values to the number of buckets - 1
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, self.num_buckets - 1, device=self.device))

    return torch.where(is_small, masked_rel_pos, val_if_large)

  def forward(self, block_size):
    block_pos = torch.arange(block_size, dtype=torch.long, device=self.device)
    context_pos = torch.arange(-block_size, block_size, dtype=torch.long, device=self.device) # XL memory, context is twice block size, and current position starts in the middle.
    block_rel_pos = rearrange(block_pos, 'i -> i 1')
    context_rel_pos = rearrange(context_pos, 'j -> 1 j')
    rel_pos = context_rel_pos - block_rel_pos

    position_bucket_indices = self.relative_position_bucket(rel_pos)

    rp_values = self.relative_attention_embedding(position_bucket_indices)
    rp_values = rearrange(rp_values, 'i j h -> () h i j')
    
    return rp_values * self.scale
  
class XLAttention(torch.nn.Module):

    def __init__(self, n_embed, n_head, dropout, device):
        super().__init__()
        self.n_embed = n_embed
        self.n_head = n_head
        self.head_size = n_embed // n_head
        self.device = device
        head_total_size = n_head * self.head_size
        self.key = torch.nn.Linear(n_embed, head_total_size, bias=False)
        self.query = torch.nn.Linear(n_embed, head_total_size, bias=False)
        self.value = torch.nn.Linear(n_embed, head_total_size, bias=False)
        self.project = torch.nn.Linear(head_total_size, n_embed)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, relative_positions, x, xl_memory):

        B, T, C = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Chris's implementation does `queries = queries * (self.head_size ** -0.5)` here but I don't think it is correct.

        k_xl, v_xl = xl_memory.unbind(dim = -2) # assume stacked
        k = torch.cat((k_xl, k), dim = -2) # prepend XL memory
        v = torch.cat((v_xl, v), dim = -2) # prepend XL memory

        ### LOCAL ATTENTION

        # Split heads
        q = rearrange(q, 'b t (h d) -> b h t d', h = self.n_head)
        k = rearrange(k, 'b t (h d) -> b h t d', h = self.n_head)
        v = rearrange(v, 'b t (h d) -> b h t d', h = self.n_head)

        w = einsum(q, k, 'b h i d, b h j d -> b h i j')
        i, j = w.shape[-2:]

        # Add relative positional encoding and scale
        w = w + relative_positions[..., -i:, -j:]
        w = w * (self.head_size ** -0.5)
        
        mask = torch.ones((i,j), dtype = torch.bool, device=self.device).triu(j-i+1) # Can't cache this as its shape depends on whether we have XL memory or not.
        w = w.masked_fill(mask, float('-inf'))

        self.dropout(w)
        w = F.softmax(w, dim=-1)

        weighted_values = w@v # b h t d
        # Concat heads
        weighted_values = rearrange(weighted_values, 'b h t d -> b t (h d)')
        
        out = self.project(weighted_values)

        # new XL memories

        # Concatenate key and value heads
        k = rearrange(k, 'b h t d -> b t (h d)', h = self.n_head)
        v = rearrange(v, 'b h t d -> b t (h d)', h = self.n_head)
        current_kv = torch.stack((k, v), dim=-2) # b t 2 (h d)

        new_xl_memory = current_kv[:, -T:]

        return self.dropout(out), new_xl_memory
    
class KNN_XLAttention(torch.nn.Module):

    def __init__(self, sample_length, max_file_length, top_k, n_embed, n_head, dropout, device):
        super().__init__()
        self.n_embed = n_embed
        self.top_k = top_k
        self.n_head = n_head
        head_size = n_embed // n_head
        self.scale_factor = head_size ** -0.5
        self.key = torch.nn.Linear(n_embed, n_embed, bias=False)
        self.query = torch.nn.Linear(n_embed, n_embed, bias=False)
        self.value = torch.nn.Linear(n_embed, n_embed, bias=False)
        self.project = torch.nn.Linear(n_embed, n_embed)
        self.dropout = torch.nn.Dropout(dropout)
        self.device = device

        # Memory per batch dim, e.g. 32 sequences at 256 per sequence is 8192 memories per batch dim.
        self.max_memories = max_file_length * sample_length
        self.knn = None
        self.current_file_idxs = None

        # Could print the gate bias as it is only 1 layer of head dim (but what is a good value?)
        self.gate_bias = torch.nn.Parameter(torch.randn(self.n_head, 1, 1))
    
    def clear_memory(self):
        if self.knn != None:
            for knn in self.knn.values():
                knn.clear()
        self.knn = None
        self.current_file_idxs = None

    def forward(self, batch_file_idxs, relative_positions, x, xl_memory, inference_mode=False):

        B, T, C = x.shape

        if self.knn is None:
            self.knn = {i: KNN(dim=self.n_embed, max_memories=self.max_memories, device=self.device) for i in range(B)}

        # Clear batch dim's knn memory if file changes
        if self.current_file_idxs != None:
            for i in range(B):
                if self.current_file_idxs[i] != batch_file_idxs[i]:
                    # print(f'Clearing knn memory for batch dim {i}')
                    self.knn[i].clear()

        self.current_file_idxs = batch_file_idxs

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # This helps to mitigate drift in the embeddings which can cause the historical keys to become less aligned to the current queries.
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        k_xl, v_xl = xl_memory.unbind(dim = -2) # assume stacked
        k = torch.cat((k_xl, k), dim = -2) # prepend XL memory
        v = torch.cat((v_xl, v), dim = -2) # prepend XL memory

        ### LOCAL ATTENTION

        # Split heads
        q = rearrange(q, 'b t (h d) -> b h t d', h = self.n_head)
        k = rearrange(k, 'b t (h d) -> b h t d', h = self.n_head)
        v = rearrange(v, 'b t (h d) -> b h t d', h = self.n_head)

        w = einsum(q, k, 'b h i d, b h j d -> b h i j')
        i, j = w.shape[-2:]
        
        # Add relative positional encoding and scale

        w = w + relative_positions[..., -i:, -j:]
        w = w * self.scale_factor

        mask = torch.ones((i,j), dtype = torch.bool, device=self.device).triu(j-i+1) # Can't cache this as its shape depends on whether we have XL memory or not.
        w = w.masked_fill(mask, float('-inf'))
        
        self.dropout(w)
        w = F.softmax(w, dim=-1)

        weighted_values = w@v # b h t d

        ### KNN ATTENTION
        knn_mask = torch.tensor([self.knn[i].index.ntotal >= self.top_k for i in range(B)], dtype=torch.bool, device=self.device)

        # Only do knn if there are at least some memories
        if knn_mask.any():

            # t1 = time.time()
            # print ("Begin KNN operations")

            # Convert queries to search form
            q = rearrange(q, 'b h t d -> b t (h d)')

            mem_kv = torch.stack([self.knn[i].search(q[i], top_k=self.top_k) for i in range(B)], dim = 0) # b, t, k, 2, c
            
            mem_k, mem_v = mem_kv.unbind(dim = -2)
            mem_k = rearrange(mem_k, 'b t k (h d) -> b h t k d', h=self.n_head)
            mem_v = rearrange(mem_v, 'b t k (h d) -> b h t k d', h=self.n_head)

            # Convert queries to attention form
            q = rearrange(q, 'b t (h d) -> b h t d', h = self.n_head)

            # Sum over d for each combination of batch, head, time and top k to get qk affinities, and hence weights for each k. resulting in a tensor of shape (b, h, t, k).
            mem_w = einsum(q, mem_k, 'b h t d, b h t k d -> b h t k')
            mem_w = mem_w * self.scale_factor

            self.dropout(mem_w)
            mem_w = F.softmax(mem_w, dim=-1)

            # Weighted sum over the top k dimension for each combination of b, h, and t, resulting in a tensor of shape (b, h, t, d). Equivalent to doing w@v for each k and summing.
            mem_weighted_values = einsum(mem_w, mem_v, 'b h t k, b h t k d -> b h t d')

            ## Combined attention
            
            # Assume every memory has content. Empty memories will be masked out below.
            combined_weighted_values = mem_weighted_values * self.gate_bias + weighted_values * (1 - self.gate_bias)

            # Mask out combined weighted values where knn memory *is* empty and non-combined values where it *is not* empty, then merge them.
            combined_weighted_values = combined_weighted_values * knn_mask.view(B, 1, 1, 1) + weighted_values * (~knn_mask).view(B, 1, 1, 1)

            # Concat heads
            combined_weighted_values = rearrange(combined_weighted_values, 'b h t d -> b t (h d)')
            out = self.project(combined_weighted_values)

            # t2 = time.time()
            # print ("End KNN operations, time taken:", t2-t1)

        else:
            # Concat heads
            weighted_values = rearrange(weighted_values, 'b h t d -> b t (h d)')
            out = self.project(weighted_values)


        # New XL memories

        # Concatenate key and value heads
        k = rearrange(k, 'b h t d -> b t (h d)', h = self.n_head)
        v = rearrange(v, 'b h t d -> b t (h d)', h = self.n_head)
        current_kv = torch.stack((k, v), dim=-2) # b t 2 c

        new_xl_memory = current_kv[:, -T:]
        # print(f'new mem shape:{new_xl_memory.shape}')
        for i in range(B):
            if inference_mode:
                # During inference, we advance one token at a time.
                trimmed_xl_memory = new_xl_memory[i][-1].unsqueeze(0)
                self.knn[i].add(trimmed_xl_memory)
            else:
                # During training, we advance a whole sequence block at a time.
                self.knn[i].add(new_xl_memory[i])

        return self.dropout(out), new_xl_memory
    
class FeedForward(torch.nn.Module):

    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_embed, 4 * n_embed), # 4x is a common expansion factor
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(4 * n_embed, n_embed) # Project back to the residual stream
        )

    def forward(self, x):
        return self.net(x)
    
class Block(torch.nn.Module):

    def __init__(self, n_embed, n_head, dropout, device):
        super().__init__()
        self.attention = XLAttention(
                            n_embed = n_embed,
                            n_head = n_head,
                            dropout = dropout, 
                            device = device)
        self.ff = FeedForward(n_embed, dropout)
        self.layer_norm1 = torch.nn.LayerNorm(n_embed)
        self.layer_norm2 = torch.nn.LayerNorm(n_embed)

    def forward(self, rel_pos, x, xl_memories):
        # Residual connections
        attn_out, new_xl_memories = self.attention(relative_positions=rel_pos, x=self.layer_norm1(x), xl_memory=xl_memories)
        x = x + attn_out
        x = x + self.ff(self.layer_norm2(x))
        return x, new_xl_memories
    
class KNNBlock(torch.nn.Module):

    def __init__(self, sample_length, max_file_length, n_embed, n_head, top_k, dropout, device):
        super().__init__()
        self.attention = KNN_XLAttention(
                            sample_length = sample_length, 
                            max_file_length = max_file_length,
                            top_k= top_k,
                            n_embed = n_embed,
                            n_head = n_head,
                            dropout = dropout,
                            device = device)
        self.ff = FeedForward(n_embed, dropout)
        self.layer_norm1 = torch.nn.LayerNorm(n_embed)
        self.layer_norm2 = torch.nn.LayerNorm(n_embed)

    def forward(self, batch_file_idxs, relative_positions, x, xl_memory = None, inference_mode=False):
        # Residual connections
        attn_out, new_xl_memories = self.attention(batch_file_idxs=batch_file_idxs, relative_positions=relative_positions, x=self.layer_norm1(x), xl_memory=xl_memory, inference_mode=inference_mode)
        x = x + attn_out
        x = x + self.ff(self.layer_norm2(x))
        return x, new_xl_memories
    
class DecoderTransformer_KNN_XL(torch.nn.Module):

    def __init__(
            self,
            vocab,
            sample_length,
            max_file_length,
            device,
            use_knn = False,
            n_embed = 512, # /8 heads = 64 per head
            n_head = 8, 
            n_layer = 8, 
            max_bar_position = 1024,
            top_k = 16,
            dropout = 0.2,
            n_rel_pos_buckets = 32,
            rel_pos_max_distance = 128):
        
        super().__init__()
        self.vocab = vocab
        self.n_embed = n_embed
        head_size = n_embed // n_head
        scaling_factor = head_size ** 0.5
        self.sample_length = sample_length
        self.device = device
        self.use_knn = use_knn
        self.n_layer = n_layer
        self.max_bar_position = max_bar_position
        self.current_file_idxs = None
        self.token_embedding = torch.nn.Embedding(self.vocab.size, n_embed)
        self.rel_pos = XLRelativePosition(n_buckets = n_rel_pos_buckets, max_distance = rel_pos_max_distance, n_head = n_head, scaling_factor = scaling_factor, device = device)
        self.rel_pos_knn = XLRelativePosition(n_buckets = n_rel_pos_buckets, max_distance = rel_pos_max_distance, n_head = n_head, scaling_factor = scaling_factor, device = device)
        self.beat_embedding = torch.nn.Embedding(SAMPLES_PER_BAR, n_embed)
        self.bar_embedding = torch.nn.Embedding(max_bar_position, n_embed)
        
        self.blocks = torch.nn.ModuleList([])
        for i in range(n_layer): # 0 -> (n_layer - 1)

            if self.isKNNLayer(i):
                self.blocks.append(KNNBlock(sample_length=sample_length, max_file_length=max_file_length, n_embed=n_embed, n_head=n_head, top_k=top_k, dropout=dropout, device=device))
            else:
                self.blocks.append(Block(n_embed=n_embed, n_head=n_head, dropout=dropout, device=device))
            
        self.layer_norm = torch.nn.LayerNorm(n_embed)
        self.lm_head = torch.nn.Linear(n_embed, self.vocab.size)

    def isKNNLayer(self, i):
        if self.use_knn:
            return (i+1) == self.n_layer - 3 # zero based index
        else:
            return False

    def forward(self, batch_file_idxs, x, xl_memories=None, targets=None):

        B, T, C = x.shape

        inference_mode = targets is None
        first_pass = xl_memories is None

        # Could split these out in one go using the unbind function
        token_idx = x[:, :, 0] # (B,T)
        time_idx = x[:, :, 1] # (B,T)

        sample_idx = time_idx % SAMPLES_PER_BAR # (B,T)
        bar_idx = (time_idx // SAMPLES_PER_BAR) % self.max_bar_position # (B,T)

        rel_pos = self.rel_pos(T)
        rel_pos_knn = self.rel_pos_knn(T)

        token_embed = self.token_embedding(token_idx) # (B,T,Embed)
        bar_embed = self.bar_embedding(bar_idx) # (B,T,Embed)
        sample_embed = self.beat_embedding(sample_idx) # (B,T,Embed)

        x = token_embed + bar_embed + sample_embed

        # If no XL memories, initialise them as 0 and reset the KNN memory
        if xl_memories is None:
            self.current_file_idxs = None
            xl_memories = []
            empty_batch_mem = torch.zeros((B, T, 2, self.n_embed), dtype=torch.long, device=self.device)
            for layer, block in enumerate(self.blocks):
                xl_memories.append(empty_batch_mem.detach().clone())
                if self.isKNNLayer(layer):
                    block.attention.clear_memory()

        # If any file has changed (and it isn't the first run), replace the XL memory for that specific batch dim in every layer with 0
        if self.current_file_idxs != None:
            empty_mem = torch.zeros((T, 2, self.n_embed), dtype=torch.long, device=self.device)
            for batch_dim, current_file_idx in enumerate(self.current_file_idxs):
                if current_file_idx != batch_file_idxs[batch_dim]:
                    # print(f"Clearing XL mem for batch dim {batch_dim}")
                    for layer in range(self.n_layer):
                        xl_memories[layer][batch_dim] = empty_mem.detach().clone()
        
        self.current_file_idxs = batch_file_idxs

        # Store the XL memories for each pass
        new_xl_memories = []

        for i, block in enumerate(self.blocks):
            if self.isKNNLayer(i):
                x, xl_mem = block(batch_file_idxs, rel_pos_knn, x, xl_memories[i], inference_mode=(inference_mode and not first_pass))
            else:
                x, xl_mem = block(rel_pos, x, xl_memories[i])

            new_xl_memories.append(xl_mem.detach())

        x = self.layer_norm(x)

        # TODO: Convert this section to use einops rearrange
        if inference_mode:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        else:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            B, T, C = logits.shape
            logits = logits.reshape(B*T, C) # Flatten all the batches
            targets = targets.reshape(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss, new_xl_memories

    @torch.no_grad()
    def generate(self, x, max_new_tokens=1024, temperature=1.0):
        self.eval()
        
        B, T, C = x.size()
        
        # We will just have one 'file' per dimension we are generating so that the knn memory is created and persists for the whole generation.
        file_idxs = torch.arange(B, device=self.device)
        xl_memories = None
        dur_start, _ = self.vocab.duration_range
        
        for _ in range(max_new_tokens):

            # Get the second to last note index if it exists, otherwise return pad idx
            if x.size(1) > 1:
                second_to_last_nidx = x[:, -2, 0].unsqueeze(0) # (B, 1)
            else:
                second_to_last_nidx = torch.stack([torch.tensor([self.vocab.pad_idx], device=self.device) for _ in range(B)], dim=0)

            # print(f'second_to_last_nidx: {second_to_last_nidx.size()}')
            
            # Could probably use unbind here
            last_nidx = x[:, -1, 0] # (B, 1)
            # print(f'last_nidx: {last_nidx.size()}')
            last_tidx = x[:, -1, 1] # (B, 1)
            # print(f'last_tidx: {last_tidx.size()}')

            # If two tokens ago was a separator, the last token was a time-incrementing duration
            duration_mask = second_to_last_nidx == self.vocab.sep_idx # (B, 1)

            # Offset the duration idx to get the actual duration, and zero out if the previous token was not a separator
            t_inc = (last_nidx - dur_start) * duration_mask
            # print(f't_inc: {t_inc.size()}')

            # Increment the time index by the duration
            tidx_next = last_tidx + t_inc # (B, 1)
            # print(f'tidx: {tidx_next.size()}')

            # if the sequence context is growing too long we must crop it at block_size
            x_cropped = x if x.size(1) <= self.sample_length else x[:, -self.sample_length:]

            # forward the model to get the logits for the index in the sequence
            logits, _, xl_memories = self(file_idxs, x_cropped, xl_memories)

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)

            # sample from the distribution
            nidx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # print(f'nidx: {nidx_next.size()}')

            # Concat with the time index
            idx_next = torch.cat((nidx_next, tidx_next), dim=1).unsqueeze(0) # (B, C)
            # print(f'idx_next: {idx_next.size()}')

            # append sampled index to the running sequence and continue
            x = torch.cat((x, idx_next), dim=1) # (B, T+1, C)

        self.train()
        return x