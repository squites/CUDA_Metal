__global__ void attention_simple(
    float* Q,      
    float* K,      
    float* V,      
    float* output,
    int seq_len,
    int head_dim
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    float scale = 1.0 / sqrtf(head_dim);
    
    float max_score = -1e30;
    for (int j = 0; j < seq_len; j++) {
        float score = 0.0;
        for (int d = 0; d < head_dim; d++) {
            score += Q[row * head_dim + d] * K[j * head_dim + d];
        }
        score *= scale;
        if (score > max_score) { 
            max_score = score;
        }
    }
    
    float sum = 0.0;
    for (int j = 0; j < seq_len; j++) {
        float score = 0.0;
        for (int d = 0; d < head_dim; d++) {
            score += Q[row * head_dim + d] * K[j * head_dim + d];
        }
        score *= scale;
        sum += expf(score - max_score);
    }
    
    for (int d = 0; d < head_dim; d++) {
        float out = 0.0;
        for (int j = 0; j < seq_len; j++) {

            float score = 0.0;
            for (int k = 0; k < head_dim; k++) {
                score += Q[row * head_dim + k] * K[j * head_dim + k];
            }
            score *= scale;
            float attn_weight = expf(score - max_score) / sum;
            
            out += attn_weight * V[j * head_dim + d];
        }
        output[row * head_dim + d] = out;
    }
}