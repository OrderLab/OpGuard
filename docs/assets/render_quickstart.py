"""Render crisp, typography-first README GIF concepts with Pillow.

Run without arguments to produce light and dark review variants. Use --primary NAME only
when a design is selected to replace quickstart.gif. Fonts stay on the host.
"""
import argparse
from functools import lru_cache
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageColor

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
            self.text(x, 498, f'0{i+1}  {label}', size=14,
                      color='accent' if i == step else 'muted')


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


def source_backdrop(c, reveal, highlight):
    """Reflowed buffer-view excerpt; softly reveal storage and its write."""
    def tint(color, opacity):
        bg = ImageColor.getrgb(c.t['bg'])
        fg = ImageColor.getrgb(c.t[color])
        return tuple(round(a + (b-a)*opacity) for a, b in zip(bg, fg))

    rows = [
        'if self.contiguous_gradients:',
        '    new_grad_tensor = (',
        '        self.ipg_buffer[self.ipg_index]',
        '        .narrow(',
        '            0,',
        '            self.elements_in_ipg_bucket,',
        '            param.numel()',
        '        )',
        '    )',
        '    new_grad_tensor.copy_(grad_reduc.view(-1))',
        '    grad_reduc.data = new_grad_tensor.data',
    ]
    x, y, size, leading = 495, 188, 15, 25
    for i, row in enumerate(rows):
        yy = layout_y(y+i*leading)*SCALE
        token = ('self.ipg_buffer[self.ipg_index]' if i == 2 else
                 'new_grad_tensor.copy_' if i == 9 else None)
        if token:
            prefix = row[:row.index(token)]
            xx = x*SCALE+c.d.textlength(prefix, font=font(size, mono=True))
            tw = c.d.textlength(token, font=font(size, mono=True))
            c.d.rounded_rectangle((xx-4*SCALE, yy-4*SCALE, xx+tw+4*SCALE, yy+(size+5)*SCALE),
                                     radius=4*SCALE, fill=tint('accent', .09*highlight))
        c.d.text((x*SCALE, yy), row, font=font(size, mono=True),
                 fill=tint('muted', .34*reveal), anchor='lt')
        if token:
            c.d.text((xx, yy), token, font=font(size, mono=True),
                     fill=tint('accent', .34*reveal+.56*highlight), anchor='lt')


def standard(theme, step, count=None, reveal=1, highlight=1):
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
        source_backdrop(c, reveal, highlight)
        c.text(40, 109, 'Root-cause analysis', size=18, color='muted')
        c.text(40, 205, 'Likely root cause', size=16, color='accent')
        c.text(40, 242, 'Gradient buffer reused', size=34, bold=True)
        c.text(40, 287, 'before reduction', size=34, bold=True)
        c.text(40, 332, 'completes.', size=34, bold=True)
        c.text(40, 403, 'Perfetto · vector_norm#2144 · rank 0', size=16, color='muted')
        c.text(40, 430, 'stage_1_and_2.py:967–1004', size=16, color='muted')
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
    frames.extend([indexed(stills[1]), indexed(standard(name, 2, reveal=0, highlight=0))])
    durations.extend([2400, 500])
    def ease(t):
        return t*t*(3-2*t)
    for i in range(1, 17):
        frames.append(indexed(standard(name, 2, reveal=ease(i/16), highlight=0)))
        durations.append(60)
    for i in range(1, 13):
        frames.append(indexed(standard(name, 2, highlight=ease(i/12))))
        durations.append(60)
    frames.append(indexed(stills[2]))
    durations.append(6820)
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
