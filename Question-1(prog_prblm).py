import numpy as np

def scaled_dot_product_attention(Q, K, V):
    """
    Computes scaled dot-product attention.
    Q, K, V are numpy arrays with shapes:
    Q: (n_queries, d_k)
    K: (n_keys, d_k)
    V: (n_keys, d_v)
    Returns:
        attention_weights: (n_queries, n_keys)
        context_vector: (n_queries, d_v)
    """
    
    d_k = K.shape[-1]                                  # key dimension
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)          # QK^T / sqrt(d_k)
    
    # Softmax across keys axis
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    context_vector = np.matmul(attention_weights, V)   # Weighted sum of V
    return attention_weights, context_vector
