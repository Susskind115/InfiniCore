#ifndef __FLASH_ATTENTION_ONLINE_SOFTMAX_CUH__
#define __FLASH_ATTENTION_ONLINE_SOFTMAX_CUH__

#include <cuda_runtime.h>

namespace op::flash_attention::cuda {

struct OnlineSoftmaxState {
    float m = -INFINITY;
    float l = 0.0f;

    __device__ __forceinline__ void update(float x, float &alpha, float &beta) {
        const float m_new = fmaxf(m, x);
        alpha = expf(m - m_new);
        beta = expf(x - m_new);
        l = l * alpha + beta;
        m = m_new;
    }
};

} // namespace op::flash_attention::cuda

#endif // __FLASH_ATTENTION_ONLINE_SOFTMAX_CUH__

