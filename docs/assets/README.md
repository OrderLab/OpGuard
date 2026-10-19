# README animation

The animation is an illustrated guide to installation, an agent request, and
reading a source-supported root-cause analysis. It is not a recording of an autonomous
agent session. The final report uses the public DeepSpeed overlap-buffer case documented in
[harness/examples/demo_case5](../../harness/examples/demo_case5/README.md): the
first observed norm-output difference leads to a source-derived buffer-reuse race.
It cites the Perfetto pivot and the embedded source snapshot. The trace alone does
not prove the race, so the report labels it a likely cause and proposes a targeted
validation rather than claiming that a GPU intervention succeeded.

The README uses `paper` in light mode and `terminal` in dark mode, both without a
top masthead and with compact section titles. A `<picture>` element selects the GIF using `prefers-color-scheme`;
`quickstart.gif` is the fallback for clients without responsive-image support.

## Review variants

Run with Pillow installed (a documentation-only dependency):

```bash
python docs/assets/render_quickstart.py
```

Outputs are in `quickstart-variants/`:

- `paper.gif`: light background, blue accent, full-width content.
- `terminal.gif`: dark background, mint accent, full-width content.

Each design is 1920 × 880, intended to display at 960 × 440 or smaller, with
three scenes over roughly 18 seconds. The second scene types a natural-language
request in a clearly labeled Codex / Claude Code conversation; the report holds
for nine seconds so its evidence references can be read. A shared palette and no dithering
keep letter edges consistent. PNG previews include the final scene and each step.
Fonts are loaded from the host (Arial/Andale Mono on macOS, DejaVu on Linux);
font files are not redistributed. Exact raster output can vary with host fonts.

Regenerate the selected README image:

```bash
python docs/assets/render_quickstart.py --readme
```

The review variants use current source-install commands from a cloned repository.
They make no timing, automatic-fix or complete-root-cause claim. Generating review
variants alone does not overwrite `quickstart.gif`.
