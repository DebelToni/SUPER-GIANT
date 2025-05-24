import numpy as np

def data_loader(dataset_tokens, batch_size):
    # dataset_tokens is a list or array of shape [num_samples, seq_len]
    num_samples = len(dataset_tokens)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    for start in range(0, num_samples, batch_size):
        batch_idx = indices[start : start+batch_size]
        batch_seq = dataset_tokens[batch_idx]  # shape [batch, seq_len]
        # Here dataset_tokens could be a numpy array of int32 already.
        batch_seq = np.array(batch_seq, dtype=np.int32)
        # inputs and targets:
        inputs = batch_seq[:, :-1]   # all tokens except last
        targets = batch_seq[:, 1:]   # all tokens except first (shifted)
        # mask for padding:
        if USE_PADDING:
            mask = (inputs != PAD_TOKEN_ID).astype(np.float32)
        else:
            mask = np.ones_like(inputs, dtype=np.float32)
        yield {'input': inputs, 'target': targets, 'mask': mask}

