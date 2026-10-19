"""python -m opguard.tools ACTION --request request.json

Exactly one JSON response on stdout; backend diagnostics go to stderr.
This CLI deliberately does not read or mutate ~/.opguard_state.yaml.
"""
import argparse
import contextlib
import json
import sys
import signal
import traceback

from . import PROTOCOL_VERSION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['capabilities', 'discover', 'prepare', 'trace', 'compare', 'align'])
    parser.add_argument('--request', help='JSON request file; omit to read stdin')
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise RuntimeError(f'Operation interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.action == 'capabilities':
            result = {'actions': ['discover', 'prepare', 'trace', 'compare', 'align'],
                      'capture_entrypoint': 'python_script', 'comparison': 'perfetto_report',
                      'numeric_verdict': False}
        else:
            with open(args.request) if args.request else contextlib.nullcontext(sys.stdin) as stream:
                request = json.load(stream)
            if request.get('protocol_version') != PROTOCOL_VERSION:
                raise ValueError('Unsupported protocol_version')
            with contextlib.redirect_stdout(sys.stderr):
                from .operations import dispatch
                result = dispatch(args.action, request)
        response = {'protocol_version': PROTOCOL_VERSION, 'ok': True, 'result': result}
        code = 0
    except (Exception, SystemExit, KeyboardInterrupt) as exc:
        traceback.print_exc(file=sys.stderr)
        response = {'protocol_version': PROTOCOL_VERSION, 'ok': False,
                    'error': {'type': type(exc).__name__, 'message': str(exc)}}
        code = 1
    print(json.dumps(response, ensure_ascii=False))
    return code


if __name__ == '__main__':
    sys.exit(main())
