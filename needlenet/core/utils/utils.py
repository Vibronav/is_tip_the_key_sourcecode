import numpy as np


def closest_divisible_by_4(n):
    q = n // 4
    n1 = q * 4
    n2 = (q + 1) * 4
    
    if abs(n - n1) < abs(n - n2):
        return n1
    else:
        return n2

def hertz_to_mel(freq_hz):
    return 2595.0 * np.log10(1.0 + freq_hz / 700.0)

def mel_to_hertz(mel):
    return 700.0 * (10**(mel / 2595.0) - 1.0)
