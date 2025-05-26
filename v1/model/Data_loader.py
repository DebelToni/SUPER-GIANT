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

        inp = batch[:, :-1].astype(np.int32)   
        tgt = batch[:, 1: ].astype(np.int32)   

        if pad_token_id is None:
            mask = np.ones_like(inp, dtype=np.float32)
        else:
            mask = (inp != pad_token_id).astype(np.float32)

        yield {"input": inp, "target": tgt, "mask": mask}

