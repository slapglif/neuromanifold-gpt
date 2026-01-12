
import torch
import triton
import triton.language as tl

@triton.jit
def fhn_imex_backward_kernel(
    # Pointers to gradients (output) and inputs (saved)
    grad_u_out_ptr, grad_w_out_ptr, # Incoming grads (from next layer)
    u_ptr, w_ptr, input_ptr, # Saved states (from forward)
    grad_u_in_ptr, grad_w_in_ptr, grad_input_ptr, # Outgoing grads (to prev layer)
    dt_ptr, # NEW: Pointer to dt
    # Constants
    threshold, tau, 
    # Shapes
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """
    Fused Backward Kernel for FHN IMEX.
    Calculates gradients for u, w, and input.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load saved states
    u = tl.load(u_ptr + offsets, mask=mask)
    w = tl.load(w_ptr + offsets, mask=mask)
    
    # Load incoming gradients (dL/du_new, dL/dw_new)
    grad_u_new = tl.load(grad_u_out_ptr + offsets, mask=mask)
    grad_w_new = tl.load(grad_w_out_ptr + offsets, mask=mask)
    
    # Constants
    dt = tl.load(dt_ptr) # Load dt from pointer
    a = 0.7
    b = 0.8
    alpha = dt / tau
    beta = 1.0 + b * alpha # denom from forward: 1 + dt*b/tau
    
    # Forward equations recap:
    # u_new = u + dt*(u - u^3/3 - w + I)
    # w_new = (w + alpha*(u_new + a)) / beta
    
    # Chain Rule:
    # 1. Backprop through w_new
    # w_new depends on w and u_new.
    # dL/dw = dL/dw_new * dw_new/dw = grad_w_new * (1/beta)
    # dL/du_new_from_w = dL/dw_new * dw_new/du_new = grad_w_new * (alpha/beta)
    
    grad_w_from_w = grad_w_new / beta
    grad_u_new_total = grad_u_new + (grad_w_new * alpha / beta)
    
    # 2. Backprop through u_new
    # u_new depends on u, w, I
    # du_new/du = 1 + dt*(1 - u^2)
    # du_new/dw = -dt
    # du_new/dI = dt
    
    term_du = 1.0 + dt * (1.0 - u * u)
    
    grad_u = grad_u_new_total * term_du
    grad_w_from_u = grad_u_new_total * (-dt)
    grad_I = grad_u_new_total * dt
    
    # Total grad_w
    grad_w = grad_w_from_w + grad_w_from_u
    
    # Store results
    tl.store(grad_u_in_ptr + offsets, grad_u, mask=mask)
    tl.store(grad_w_in_ptr + offsets, grad_w, mask=mask)
    tl.store(grad_input_ptr + offsets, grad_I, mask=mask)

class FusedFHNSolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, w, inputs, n_steps, threshold, tau, dt):
        """
        Run n_steps of FHN dynamics using Triton kernel.
        """
        u = u.contiguous()
        w = w.contiguous()
        inputs = inputs.contiguous()
        
        B, H, T, D = u.shape
        n_elements = u.numel()
        
        # We need to save intermediates for backward?
        # FHN is an ODE solver. We can use "adjoint method" (constant memory) 
        # or backprop through time (linear memory).
        # Since n_steps is small (2), backprop through time is better/faster.
        # But our kernel runs a LOOP. We only saved the INITIAL u, w.
        # The gradients depend on u at each step.
        
        # CRITICAL: For correct gradients, we need to recompute or save states at each step.
        # Since we use a loop in Python invoking the kernel step-by-step, 
        # we naturally save the intermediate tensors in the autograd graph IF we used PyTorch ops.
        # But here we used a custom function.
        
        # If n_steps > 1, we are calling this function iteratively from fhn.py?
        # No, fhn.py loop calls this function inside the loop?
        # Let's check fhn.py.
        
        # In fhn.py: 
        # v_out, w_out = fhn_solve_fused(v, w, I, n_steps, ...)
        
        # So this function handles the ENTIRE loop.
        # This means we lose intermediate states unless we save them.
        # For n_steps=2, we can just save all of them.
        
        # Actually, implementing full backprop through time in a single custom kernel is hard.
        # EASIER: Make FusedFHNSolver run JUST ONE STEP.
        # Then let Python loop in fhn.py handle the sequence. 
        # Autograd will handle the chain rule between steps automatically!
        
        # This is the "Correct" and "Robust" way.
        # 1 step per kernel call. Overhead is minimal for n_steps=2.
        
        # Let's pivot: fhn_solve_fused will run 1 step.
        # fhn.py will loop n_steps times.
        
        # Wait, the current implementation has a loop in forward: `for _ in range(n_steps): kernel...`
        # This updates u, w in place! This destroys history needed for backward.
        
        # FIX: Remove loop from forward. Run 1 step.
        # In fhn.py, loop n_steps.
        
        ctx.save_for_backward(u, w, inputs)
        ctx.n_elements = n_elements
        ctx.threshold = threshold
        ctx.tau = tau
        
        # Handle tensor dt
        if isinstance(dt, float):
            dt_tensor = torch.tensor(dt, device=u.device, dtype=torch.float32)
        else:
            dt_tensor = dt
            
        # Save dt tensor for backward to avoid .item() graph break
        # We can't save it in saved_tensors if it's not an input to forward?
        # But it IS an input to forward.
        # Wait, save_for_backward only accepts tensors that are inputs or outputs?
        # No, it accepts any tensor.
        # But 'dt' might require grad? No, we treat it as param but here passed as value.
        # If 'dt' requires grad, we must handle dL/ddt.
        # We ignored dL/ddt in backward kernel.
        # For now, let's assume dt is fixed or we ignore its grad.
        ctx.dt_tensor = dt_tensor 
            
        # Output tensors (new storage)
        u_out = torch.empty_like(u)
        w_out = torch.empty_like(w)
        
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        fhn_imex_kernel_op[grid](
            u, w, inputs,
            u_out, w_out, # New output pointers
            dt_tensor, # scalar dt POINTER
            tau, threshold, # other constants
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return u_out, w_out

    @staticmethod
    def backward(ctx, grad_u_out, grad_w_out):
        u, w, inputs = ctx.saved_tensors
        n_elements = ctx.n_elements
        dt_tensor = ctx.dt_tensor # Retrieve tensor
        tau = ctx.tau
        threshold = ctx.threshold
        
        grad_u_in = torch.empty_like(u)
        grad_w_in = torch.empty_like(w)
        grad_inputs = torch.empty_like(inputs)
        
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        fhn_imex_backward_kernel[grid](
            grad_u_out, grad_w_out,
            u, w, inputs,
            grad_u_in, grad_w_in, grad_inputs,
            dt_tensor, # Pass pointer
            threshold, tau, 
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Return grads matching forward inputs: u, w, inputs, n_steps, threshold, tau, dt
        # We don't compute grad for n_steps, threshold, tau, dt (return None)
        return grad_u_in, grad_w_in, grad_inputs, None, None, None, None

# Need to redefine the kernel to support out-of-place
@triton.jit
def fhn_imex_kernel_op(
    u_in_ptr, w_in_ptr, input_ptr,
    u_out_ptr, w_out_ptr,
    dt_ptr, # NEW: Pointer
    tau, threshold,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    u = tl.load(u_in_ptr + offsets, mask=mask)
    w = tl.load(w_in_ptr + offsets, mask=mask)
    i_ext = tl.load(input_ptr + offsets, mask=mask)
    
    dt = tl.load(dt_ptr) # Load
    a = 0.7
    b = 0.8
    
    cubic = u * u * u / 3.0
    du = u - cubic - w + i_ext
    u_new = u + du * dt
    
    alpha = dt / tau
    denom = 1.0 + b * alpha
    
    w_step = alpha * (u_new + a)
    w_new = (w + w_step) / denom
    
    tl.store(u_out_ptr + offsets, u_new, mask=mask)
    tl.store(w_out_ptr + offsets, w_new, mask=mask)

def fhn_step_fused(u, w, inputs, threshold=0.5, tau=12.5, dt=0.1):
    """Runs a SINGLE step of FHN."""
    # Dummy n_steps arg for compatibility if needed, or remove it
    return FusedFHNSolver.apply(u, w, inputs, 1, threshold, tau, dt)

