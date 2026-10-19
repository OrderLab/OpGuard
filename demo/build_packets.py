"""Build logs + versions versus OpGuard observations + captured source."""
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
evidence = ROOT / 'demo/evidence'
manifest = json.loads((evidence / 'manifest.json').read_text())
expected = {f['path']: f for f in manifest['files']}
source_manifest = json.loads((evidence / 'source-manifest.json').read_text())
RUNS = {'overlap-off': 'run1', 'overlap-on': 'run2'}

def neutral_labels(text):
    for side, run in RUNS.items():
        text = text.replace(side, run)
    return text

shared = ('Two training runs show numerical divergence. Diagnose it from the supplied evidence. '
          'Launch commands are not supplied. Do not assume the runs have identical configurations or that there is only one difference.\n'
          '\n--- Recorded environment (both runs) ---\n' + json.dumps(manifest['runtime'], indent=2) + '\n'
          'Dependency Git commits were not recorded; DeepSpeed reports git-hash=unknown. '
          'Package versions identify the environment, not a verified source location.\n')
source_parts = ['\n--- OpGuard captured source ---\n'
                'Line numbers refer to the captured, instrumented snapshots. Source definitions do not prove branch execution. '
                'Cite source as run1/deepspeed/runtime/zero/stage_1_and_2.py:LINE or run2/deepspeed/runtime/zero/stage_1_and_2.py:LINE.\n']
source_lines = {}
public_sources = ROOT / 'demo/sources'
public_sources.mkdir(exist_ok=True)
for source in source_manifest['sources']:
    raw = (evidence / source['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == source['sha256']
    assert source['trace_sha256'] == expected['opguard/perfetto_gpu_000.json.gz']['sha256']
    content = neutral_labels(raw.decode())
    lines = content.splitlines()
    assert len(lines) == source['lines']
    source_lines[source['run']] = lines
    label = f"{source['run']}/{source['source_file']}"
    source_parts.append(f'\n--- {label} (complete module) ---\n' + '\n'.join(f'{i}:' + (f' {line}' if line else '') for i, line in enumerate(lines, 1)) + '\n')
    (public_sources / source['path']).write_text(content)
    rows = '\n'.join(f'<span class="line" id="L{i}"><a href="#L{i}" aria-label="Line {i}">{i}</a><code>{html.escape(line)}</code></span>' for i, line in enumerate(lines, 1))
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(label)} · OpGuard source</title>
<style>body{{margin:0;font:14px/1.65 system-ui;color:#172333;background:#f7f9fc}}header{{padding:20px 28px;background:white;border-bottom:1px solid #dce2ea}}h1{{font-size:19px;margin:8px 0}}p{{margin:4px 0;color:#536179}}a{{color:#156354}}pre{{margin:0;padding:20px 0;overflow:auto}}.line{{display:block;scroll-margin-top:24px;min-width:max-content}}.line>a{{display:inline-block;text-align:right;width:5em;margin-right:20px;color:#7a8798;text-decoration:none;user-select:none}}.line:target{{background:#fff0b5}}code{{font:12px/1.65 ui-monospace,SFMono-Regular,monospace}}</style></head><body><header><a href="../../#rootcause">← OpGuard demo</a><h1>{html.escape(label)}</h1><p>Complete captured source · line numbers include instrumentation · run labels normalized</p><p>Source alone does not establish which branches executed or prove the cause.</p></header><pre>{rows}</pre></body></html>'''
    (public_sources / source['path'].replace('.py','.html')).write_text(page)
captured_source = ''.join(source_parts)
parts = ['Standard logs with normalized paths and neutral run labels, in original order within each stream. '
         'stdout/stderr were captured separately; their interleaving is unknown.\n']
for side, run in RUNS.items():
    for stream in ('stdout', 'stderr'):
        data = (evidence / f'{side}-{stream}.log').read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected[f'standard-logs/{side}/{stream}.log']['sha256']
        parts.append(f'\n--- {run} / {stream} ---\n' + neutral_labels(data.decode()))
raw = (evidence / 'opguard-output.json').read_bytes()
assert hashlib.sha256(raw).hexdigest() == expected['opguard-output.json']['sha256']
report = json.loads(raw)
limits = [neutral_labels(limit).replace('the precise racing memory operation', 'the causal operation') for limit in report['limits']]
compact = {'kind': 'first observed differences at captured callsites', 'ranks': [], 'limits': limits}
for rank in report['ranks']:
    observations = []
    for observed in rank['observations']:
        item = {k: observed[k] for k in ('run','event_ref','operation','operator_tag_line','captured_source_line','function','unaligned_before','io')}
        item['source_file'] = f"{item['run']}/deepspeed/runtime/zero/stage_1_and_2.py"
        line = item['captured_source_line']
        item['source_at_observation'] = [{'line': n, 'text': source_lines[item['run']][n-1]} for n in range(line, line+2)]
        assert str(item['operator_tag_line']) in item['source_at_observation'][0]['text']
        observations.append(item)
    compact['ranks'].append({'rank': rank['rank'], 'observations': observations})
observations = {'trace': ''.join(parts), 'opguard': json.dumps(compact, indent=2) + '\n'}
packets = {
    'trace': shared + '\n--- Runtime observations ---\n' + observations['trace'],
    'opguard': shared + captured_source + '\n--- Runtime observations ---\n' + observations['opguard'],
}
for arm, content in packets.items():
    (ROOT / f'demo/inputs/{arm}.txt').write_text(content)
locations = [{'rank': rank['rank'], 'source_file': observed['source_file'], 'line': observed['captured_source_line']}
             for rank in compact['ranks'] for observed in rank['observations']]
(ROOT / 'demo/packets.js').write_text('// Generated by build_packets.py.\nexport const packets = ' + json.dumps(packets) + ';\n'
                                   'export const capturedLocations = ' + json.dumps(locations) + ';\n')
(ROOT / 'demo/evidence-stats.json').write_text(json.dumps({'source_bytes':{'trace':0,'opguard':len(captured_source.encode())},'observation_bytes':{k:len(v.encode()) for k,v in observations.items()},'packet_bytes':{k:len(v.encode()) for k,v in packets.items()},'source_lines':{k:len(v) for k,v in source_lines.items()}},indent=2)+'\n')
print({k: len(v.encode()) for k,v in packets.items()})
