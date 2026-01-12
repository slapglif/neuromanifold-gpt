
import torch
import torch.nn.functional as F

def parallel_fhn_scan(u_init, w_init, I, a, b, tau, dt):
    """
    Parallel scan implementation of linearized FHN dynamics.
    
    FHN Linear Dynamics (Piecewise Linear Approximation):
    du = u - w + I
    dw = (u - b*w + a) / tau
    
    State matrix A:
    [ 1+dt      -dt       ]
    [ dt/tau    1-b*dt/tau]
    
    State h = [u, w].
    h_t = A * h_{t-1} + B * I_t
    
    This is a Linear Recurrence (SSM). We can use parallel scan (cumsum in log-space).
    
    Args:
        u_init, w_init: Initial states
        I: (B, T, D) Input current
        a, b, tau, dt: Parameters
    """
    B, T, D = I.shape
    
    # 1. Construct State Transition Matrix A (2x2)
    # Actually, A depends on parameters.
    # We treat u, w as independent channels for the scan if they were decoupled,
    # but they are coupled.
    
    # Complex Number Trick:
    # If eigenvalues of A are complex, we can treat h as a complex number.
    # FHN oscillates -> Complex eigenvalues!
    
    # Let's use a diagonalized approximation or simple cumulative sum for the 'integration' part.
    # Simplified:
    # u_{t} = u_{t-1} + decay * u_{t-1} + input
    # u_{t} = alpha * u_{t-1} + input
    # This is solvable by parallel scan: u_t = sum(input_k * alpha^(t-k))
    
    # For FHN, the "decay" is complex (rotation).
    # We can implement this as a complex multiplication scan.
    
    # FHN Eigenvalues: lambda = -gamma +/- i*omega
    # Discrete step: z_t = z_{t-1} * exp(lambda * dt) + I_t
    
    # Calculate effective complex decay 'alpha'
    # omega (frequency) approx 1/tau? No, sqrt(1/tau).
    # gamma (damping) approx b/tau?
    
    # Let's assume parameters give us a complex decay alpha.
    alpha_real = 1.0 - dt # Damping
    alpha_imag = dt / tau # Rotation (coupling u->w)
    
    # Convert inputs to complex
    # I contributes to real part (u)
    I_complex = torch.complex(I, torch.zeros_like(I))
    
    # Alpha (decay/rotation factor)
    # We broadcast alpha to (B, T, D)
    alpha = torch.complex(torch.tensor(alpha_real, device=I.device), 
                          torch.tensor(alpha_imag, device=I.device))
    
    # Log-space scan for numerical stability?
    # Or just cumprod if T=256 is small enough.
    # Parallel Scan: y_t = sum_{k=0}^t x_k * prod_{j=k+1}^t alpha_j
    # With constant alpha: y_t = sum_{k=0}^t x_k * alpha^(t-k)
    # This is a convolution!
    # y = x * kernel, where kernel = [1, alpha, alpha^2, ...]
    
    # FFT Convolution Implementation
    # 1. Construct kernel
    powers = torch.arange(T, device=I.device)
    kernel = alpha.pow(powers) # (T,)
    
    # 2. Convolution (using FFT)
    # Pad to 2*T for linear convolution (avoid circular wrap)
    n_fft = 2 * T
    
    # I: (B, T, D) -> Permute to (B, D, T) for 1D FFT if needed?
    # torch.fft.fft applies to last dim by default? No, dim=-1.
    # We want FFT over T (dim 1).
    I_f = torch.fft.fft(I_complex, n=n_fft, dim=1)
    
    # Kernel: (T,) -> (2T,)
    K_f = torch.fft.fft(kernel, n=n_fft, dim=0) # (2T,)
    
    # Broadcast K_f to (B, T, D) shape
    # I_f: (B, 2T, D)
    # K_f needs to match T dim.
    # K_f.view(1, -1, 1) -> (1, 2T, 1)
    out_f = I_f * K_f.view(1, -1, 1)
    
    out = torch.fft.ifft(out_f, n=n_fft, dim=1)
    
    # Crop to T
    out = out[:, :T, :]
    
    # Extract u, w
    # u is real part
    u_out = out.real
    
    # Apply non-linearity (Thresholding/Gating) post-scan
    # Mamba style: x * silu(x)
    # FHN style: u - u^3/3
    # We approximate the "firing" by gating the output
    u_out = F.hardshrink(u_out, lambd=0.5) # Thresholding behavior
    
    # w is implicit in the complex rotation
    w_out = out.imag
    
    return u_out, w_out
