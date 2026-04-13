import numpy as np
from tests.common.common import compute_s_z, compute_sfx_zfx, dequentize_linear_fixed_point, quantize_linear_fixed_point

if __name__ == '__main__':
    Q = 31
    a = np.array([-0.2, -0.1, 0.0, 3.5, 4.0], dtype=np.float32)
    sa, za = compute_s_z(a)
    sfxa, zfxa = compute_sfx_zfx(sa, za, Q)
    aq = quantize_linear_fixed_point(a, sfxa, zfxa, Q)
    ar = dequentize_linear_fixed_point(aq, sfxa, zfxa, Q)


