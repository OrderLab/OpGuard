# opguard/wrapped_ops.py

from opguard.runtime_tracer import scoped_tensor_tracer

TRITON_GRID_KW = "__opg_triton_grid__"  # carried by CST pass 1

def make_wrapper(tag_prefix="wrapped_op", dump_tensors=False, global_sync=False):
    """
    Curried wrapper factory (lineno is REQUIRED).
    Usage *after pass 2*:
        opguard_wrapped_foo(LINENO)(original_fn, *args, **kwargs)
    
    - Receives the original function/method as the first argument.
    - Handles plain functions, bound/unbound methods, and Triton kernels.
    - Triton launches are indicated by kwarg __opg_triton_grid__=<grid>.
    """
    def inner_wrapper(lineno: int):
        @scoped_tensor_tracer(
            tag_prefix=tag_prefix,
            lineno=int(lineno),
            dump_tensors=dump_tensors,
            global_sync=global_sync,
        )
        def wrapper(original_fn, *args, **kwargs):
            # Triton path: kernel[grid](...)
            grid = kwargs.pop(TRITON_GRID_KW, None)
            if grid is not None:
                return original_fn[grid](*args, **kwargs)

            # Bound method: obj.method(...)
            if getattr(original_fn, "__self__", None) is not None:
                return original_fn(*args, **kwargs)

            # Unbound method: Type.method(obj, ...)
            if args and hasattr(original_fn, "__call__"):
                try:
                    return original_fn(args[0], *args[1:], **kwargs)
                except (TypeError, AttributeError):
                    pass  # fall through to plain function

            # Plain function
            return original_fn(*args, **kwargs)
        return wrapper
    return inner_wrapper
