import argparse
import json
import sys

from .tools import TOOLS, dispatch


def main():
    parser = argparse.ArgumentParser(description='Provider-neutral OpGuard agent tools')
    parser.add_argument('--root', default='./experiments', help='Persistent experiment store; shared with workers')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('tools', help='Print function schemas')
    call = sub.add_parser('call', help='Execute one tool and print one JSON response')
    call.add_argument('name', choices=[t['name'] for t in TOOLS])
    call.add_argument('--arguments', help='JSON file; defaults to stdin')
    args = parser.parse_args()
    if args.command == 'tools':
        print(json.dumps({'protocol_version': 1, 'tools': TOOLS}, indent=2))
        return 0
    try:
        if args.arguments:
            with open(args.arguments) as stream:
                arguments = json.load(stream)
        else:
            arguments = json.load(sys.stdin)
        result = dispatch(args.root, args.name, arguments)
        print(json.dumps({'ok': True, 'result': result}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({'ok': False, 'error': {'type': type(exc).__name__, 'message': str(exc)}}))
        return 1


if __name__ == '__main__':
    sys.exit(main())
