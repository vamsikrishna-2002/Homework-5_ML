# VAMSI KRISHNA YARRAGUNTA (700772692)  

Question-1  
Example Usage:  
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

Example Output :  
Attention Weights:  
                                 [0.4319   0.1361   0.4319]   
Context Vector:  
                                        [1.7277   0.7042]   


                              
  

Question-2  

Output Shape Verification:  
Input shape: torch.Size ([32, 10, 128])  
Output shape: torch.Size ([32, 10, 128])  

   
            
  
