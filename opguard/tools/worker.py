"""Rank entrypoint. A completion receipt is written only after final uploads.

Launch with python/torchrun -m opguard.tools.worker REQUEST SCRIPT [ARG ...].
The launcher starts with autostart disabled; only rank workers enable tracing.
"""
import atexit
import json
import os
from pathlib import Path
import runpy
import sys
import uuid


def main():
    request_file, script, *arguments = sys.argv[1:]
    request = json.loads(Path(request_file).read_text())
    success = False
    worker_pid = os.getpid()
    # Register before importing the tracer: run after its atexit callbacks.
    def finish():
        if not success or os.getpid() != worker_pid:
            return
        try:
            from opguard.distributed import get_global_rank
            from opguard.local_dump_manager import get_local_dump_manager
            rank = get_global_rank()
            manager = get_local_dump_manager()
            if manager is None:
                raise RuntimeError('No local staging manager was initialized')
            manager.close()  # Raises on final upload failure; never publish success then.
            _, owner, spool = manager._identity()
            checkpoint = spool / 'checkpoint.json'
            record = {'protocol_version': 1, 'capture_id': request['capture_id'],
                      'phase': request['phase'], 'rank': rank, 'owner': owner,
                      'checkpoint': {'sequence': json.loads(checkpoint.read_text())['sequence']} if checkpoint.exists() else None}
            receipts = Path(request['output']) / '.opguard-complete'
            receipts.mkdir(parents=True, exist_ok=True)
            # Exclusive final name detects duplicate workers/restarts with the same rank.
            temp = receipts / (uuid.uuid4().hex + '.tmp')
            temp.write_text(json.dumps(record))
            try:
                os.link(temp, receipts / f'rank-{rank}.json')
            finally:
                temp.unlink()
        except BaseException as exc:
            print(f'OpGuard finalization failed: {exc}', file=sys.stderr, flush=True)
            os._exit(1)
    atexit.register(finish)
    os.environ['OPGUARD_AUTOSTART'] = '1'
    from opguard.local_dump_manager import enable_local_dump
    enable_local_dump(request['output'], float(os.getenv('OPGUARD_LOCAL_DUMP_SYNC_INTERVAL', '1800')))
    from opguard.runtime_policy import apply_requested_policy
    apply_requested_policy()
    if request['phase'] == 'discover':
        from opguard.callsite_tracer import load_autostart_module
        autostart = load_autostart_module()
        autostart.autostart()
        if not autostart._started:
            raise RuntimeError('Callsite tracer failed to start')
    sys.argv = [script, *arguments]
    sys.path.insert(0, str(Path(script).resolve().parent))
    try:
        runpy.run_path(script, run_name='__main__')
        success = True
    except SystemExit as exc:
        success = exc.code is None or exc.code == 0
        raise


if __name__ == '__main__':
    main()
