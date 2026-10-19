import opguard.callsite_tracer.opguard_cupti as cupti
cupti.start_mem()
print("[opguard] cupti status:", cupti.get_status())
