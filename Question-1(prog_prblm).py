import numpy as np

def scaled_dot_product_attention(Q, K, V):
    """
    Computes scaled dot-product attention.
    Q, K, V shapes:
        Q: (n_queries, d_k)
        K: (n_keys, d_k)
        V: (n_keys, d_v)
    Returns:
        attention_weights: (n_queries, n_keys)
        context_vector: (n_queries, d_v)
    """

    d_k = K.shape[-1]
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)

    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    context_vector = np.matmul(attention_weights, V)
    return attention_weights, context_vector


if __name__ == "__main__":
    Q = np.array([[1.0, 0.0, 1.0]])
    K = np.array([[1.0, 2.0, 1.0],
                  [0.0, 1.0, 0.0],
                  [1.0, 0.0, 1.0]])
    V = np.array([[1.0, 0.0],
                  [0.0, 2.0],
                  [3.0, 1.0]])

    weights, context = scaled_dot_product_attention(Q, K, V)
    print("Attention Weights:\n", weights)
    print("Context Vector:\n", context)
