from .build import build_perfetto

# __init__.py
def main():
    from .cli import main as _cli_main
    _cli_main()

if __name__ == "__main__":
    main()