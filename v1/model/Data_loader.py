# import numpy as np
#
# def data_loader(dataset_tokens, batch_size):
#     # dataset_tokens is a list or array of shape [num_samples, seq_len]
#     num_samples = len(dataset_tokens)
#     indices = np.arange(num_samples)
#     np.random.shuffle(indices)
#     for start in range(0, num_samples, batch_size):
#         batch_idx = indices[start : start+batch_size]
#         batch_seq = dataset_tokens[batch_idx]  # shape [batch, seq_len]
#         # Here dataset_tokens could be a numpy array of int32 already.
#         batch_seq = np.array(batch_seq, dtype=np.int32)
#         # inputs and targets:
#         inputs = batch_seq[:, :-1]   # all tokens except last
#         targets = batch_seq[:, 1:]   # all tokens except first (shifted)
#         # mask for padding:
#         if USE_PADDING:
#             mask = (inputs != PAD_TOKEN_ID).astype(np.float32)
#         else:
#             mask = np.ones_like(inputs, dtype=np.float32)
#         yield {'input': inputs, 'target': targets, 'mask': mask}
#
# Data_loader.py
import numpy as np

def data_loader(dataset_tokens, batch_size, pad_token_id=None, shuffle=True):
    """
    dataset_tokens : np.ndarray  [N, seq_len]   already padded/truncated to context_length
    Returns dictionaries with 'input' | 'target' | 'mask'
    """
    idx = np.arange(len(dataset_tokens))
    if shuffle:
        np.random.shuffle(idx)

    for start in range(0, len(dataset_tokens), batch_size):
        sl = slice(start, start+batch_size)
        batch = dataset_tokens[idx[sl]]

        inp = batch[:, :-1].astype(np.int32)   # [B, L-1]
        tgt = batch[:, 1: ].astype(np.int32)   # [B, L-1]

        if pad_token_id is None:
            mask = np.ones_like(inp, dtype=np.float32)
        else:
            mask = (inp != pad_token_id).astype(np.float32)

        yield {"input": inp, "target": tgt, "mask": mask}

