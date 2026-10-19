"""Render crisp, typography-first README GIF concepts with Pillow.

Run without arguments to produce light and dark review variants. Use --primary NAME only
when a design is selected to replace quickstart.gif. Fonts stay on the host.
"""
import argparse
from functools import lru_cache
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
W, H, SCALE = 960, 440, 2
TOP_TRIM = 70


def layout_y(y):
    return y - TOP_TRIM - (30 if y >= 180 else 0)
THEMES = {
    'paper': dict(bg='#FAFBFC', ink='#17212B', muted='#66717F', accent='#1669D3',
                  soft='#EDF4FF', panel='#FFFFFF', line='#DFE5EC', code='#F3F5F8'),
    'terminal': dict(bg='#10151B', ink='#F1F5F9', muted='#A2AFBC', accent='#78E2C1',
                     soft='#1C332F', panel='#182029', line='#303B47', code='#111920'),

}


@lru_cache(None)
def font(size, bold=False, mono=False):
    mac = Path('/System/Library/Fonts/Supplemental')
    candidates = ([mac / 'Andale Mono.ttf', Path('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf')]
                  if mono else [mac / ('Arial Bold.ttf' if bold else 'Arial.ttf'),
                  Path('/usr/share/fonts/truetype/dejavu') / ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')])
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), round(size * SCALE))
    raise RuntimeError('Install Arial or DejaVu Sans and a monospace font before rendering.')


class Canvas:
    def __init__(self, theme):
        self.t = THEMES[theme]
        self.image = Image.new('RGB', (W * SCALE, H * SCALE), self.t['bg'])
        self.d = ImageDraw.Draw(self.image)

    def text(self, x, y, value, size=22, color='ink', bold=False, mono=False):
        f = font(size, bold, mono)
        # Anchor at the actual top rather than font-dependent baseline offsets.
        width = self.d.textlength(value, font=f) / SCALE
        if x + width > W - 20:
            raise ValueError(f'Text overflows: {value!r} ({x + width:.1f}px)')
        self.d.text((x * SCALE, layout_y(y) * SCALE), value, font=f,
                    fill=self.t.get(color, color), anchor='lt')

    def box(self, x, y, w, h, fill='panel', outline=None, radius=12):
        self.d.rounded_rectangle(tuple(round(v*SCALE) for v in (x,layout_y(y),x+w,layout_y(y)+h)),
                                 radius=radius*SCALE, fill=self.t.get(fill,fill),
                                 outline=self.t.get(outline,outline), width=SCALE)

    def line(self, x1, y1, x2, y2, color='line', width=1):
        self.d.line(tuple(round(v*SCALE) for v in (x1,layout_y(y1),x2,layout_y(y2))), fill=self.t[color], width=width*SCALE)

    def navigation(self, step):
        labels = ['Install', 'Ask your agent', 'Root-cause report']
        for i, label in enumerate(labels):
            x = 40 + i * 300
            self.line(x, 476, x+280, 476, 'accent' if i == step else 'line', 3 if i == step else 1)
            self.text(x, 492, f'0{i+1}  {label}', size=19,
                      color='accent' if i == step else 'muted', bold=i == step)


PROMPT_LINES = ('Use OpGuard to investigate these two training runs.',
                'Find the root cause and cite the Perfetto evidence.')
PROMPT_LENGTH = sum(map(len, PROMPT_LINES))


def typed_prompt(c, count):
    remaining = PROMPT_LENGTH if count is None else count
    for i, line in enumerate(PROMPT_LINES):
        shown = line[:max(0, remaining)]
        y = 284 + i * 37
        c.text(64, y, shown, size=26)
        if count is not None and 0 <= remaining < len(line):
            x = 64 + c.d.textlength(shown, font=font(26)) / SCALE
            c.box(x+3, y, 2, 27, fill='accent', radius=0)
        remaining -= len(line)


def standard(theme, step, count=None):
    c = Canvas(theme)
    c.navigation(step)
    if step == 0:
        c.text(40, 109, 'Install OpGuard', size=24, bold=True)
        c.text(40, 145, 'Install both tools from the cloned OpGuard repository.', size=18, color='muted')
        for y, label, cmd in [(215, 'Training environment', 'python -m pip install .'),
                               (331, 'Agent environment', 'uv tool install ./harness')]:
            c.box(40, y, 880, 98, fill='panel', outline='line')
            c.text(62, y+16, label, size=19, color='muted')
            c.text(62, y+52, '$', size=23, mono=True, color='accent')
            c.text(91, y+52, cmd, size=23, mono=True)
    elif step == 1:
        c.text(40, 109, 'Ask your agent', size=24, bold=True)
        c.text(40, 145, 'Use natural language in Codex or Claude Code.', size=18, color='muted')
        c.box(40, 208, 880, 240, fill='panel', outline='line')
        c.text(64, 228, 'YOU', size=17, color='accent', bold=True)
        c.text(621, 228, 'Codex  /  Claude Code', size=20, color='muted')
        c.line(64, 263, 896, 263)
        typed_prompt(c, count)
        c.text(64, 378, 'A  ./run-a/train.py     B  ./run-b/train.py', size=20, mono=True, color='muted')
        c.text(64, 415, 'Python: /path/to/train-env/bin/python', size=19, color='muted', mono=True)
    else:
        c.text(40, 109, 'Root-cause analysis', size=24, bold=True)
        c.text(40, 145, 'DeepSpeed communication overlap · evidence and next steps', size=18, color='muted')
        c.box(40, 203, 880, 250, fill='panel', outline='line')
        c.text(64, 221, 'LIKELY ROOT CAUSE', size=17, color='accent', bold=True)
        c.text(64, 253, 'Gradient buffer reused before reduction completes.', size=24, bold=True)
        c.text(64, 292, 'The overwrite can corrupt gradients consumed by the norm.', size=22)
        c.line(64, 328, 896, 328)
        c.text(64, 343, 'TRACE', size=17, color='muted', bold=True)
        c.text(150, 340, 'Perfetto / vector_norm#2144 / rank 0', size=21, color='accent')
        c.text(64, 378, 'SOURCE', size=17, color='muted', bold=True)
        c.text(150, 375, 'stage_1_and_2.py:967–1004 · buffer reuse', size=21)
        c.text(64, 412, 'NEXT', size=17, color='muted', bold=True)
        c.text(150, 409, 'Test a per-buffer completion wait before reuse.', size=21)
        c.text(40, 459, 'Public demo: source-supported diagnosis; GPU intervention not run.', size=15, color='muted')
    return c.image


def save_variant(name, directory):
    stills = [standard(name, 0),
              standard(name, 1), standard(name, 2)]
    # A shared palette keeps antialiased glyphs consistent during typing.
    palette_source = Image.new('RGB', (W*SCALE, H*SCALE*3))
    for i, frame in enumerate(stills):
        palette_source.paste(frame, (0,i*H*SCALE))
    palette = palette_source.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    def indexed(frame):
        return frame.quantize(palette=palette, dither=Image.Dither.NONE)
    frames, durations = [indexed(stills[0])], [3200]
    for count in range(0, PROMPT_LENGTH, 3):
        frames.append(indexed(standard(name, 1, count)))
        durations.append(90 if count else 450)
    frames.extend([indexed(stills[1]), indexed(stills[2])])
    durations.extend([2400, 9000])
    path = directory / f'{name}.gif'
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=False, disposal=1)
    stills[2].save(directory / f'{name}.png')
    for i, frame in enumerate(stills):
        frame.save(directory / f'{name}-step-{i+1}.png')
    print(f'{name}: {path.stat().st_size:,} bytes; {len(frames)} frames; {sum(durations)/1000:.1f}s')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['all', *THEMES], default='all')
    parser.add_argument('--primary', choices=list(THEMES), help='Also replace quickstart.gif with this design')
    parser.add_argument('--readme', action='store_true', help='Regenerate the light and dark README GIFs')
    args = parser.parse_args()
    directory = ROOT / 'quickstart-variants'
    directory.mkdir(exist_ok=True)
    names = ['paper', 'terminal'] if args.readme else (list(THEMES) if args.variant == 'all' else [args.variant])
    if not args.readme and args.primary and args.primary not in names:
        names.append(args.primary)
    for name in names:
        path = save_variant(name, directory)
        if args.readme:
            target = 'quickstart-dark.gif' if name == 'terminal' else 'quickstart.gif'
            (ROOT / target).write_bytes(path.read_bytes())
        elif name == args.primary:
            (ROOT / 'quickstart.gif').write_bytes(path.read_bytes())


if __name__ == '__main__':
    main()
