# Callsite Tracer Instrumentor

The Callsite Tracer Instrumentor automatically injects `CallsiteTracer` context managers around specified function calls based on configuration.

## Overview

This instrumentor transforms code like this:

```python
def main():
    # ... other code ...
    trainer.fit(model, datamodule)
```

Into this:

```python
from opguard.callsite_tracer.trace import CallsiteTracer

def main():
    # ... other code ...
    with CallsiteTracer(10000) as tracer:
        trainer.fit(model, datamodule)
```

## Configuration

The instrumentor reads from the `callsite_tracer` section of your configuration YAML:

```yaml
callsite_tracer:
  - file_path: "path/to/model.py"
    function: main
    traced_functions:
      - fit
      - validate
```

### Configuration Fields

- `file_path`: Path to the Python file to instrument (relative to config file)
- `function`: Name of the function containing the calls to trace
- `traced_functions`: List of function names to wrap with CallsiteTracer

## Usage

### Command Line

```bash
# Using the CLI
python -m opguard.cli inject-callsite-tracer config.yaml

# Or directly
python opguard/instrumentor/callsite_tracer_injector.py config.yaml
```

### Programmatic

```python
from opguard.instrumentor.callsite_tracer_injector import inject_callsite_tracer

inject_callsite_tracer("config.yaml")
```

## Features

1. **Automatic Import Injection**: Adds the necessary import statement if not present
2. **Context Manager Wrapping**: Wraps function calls with `CallsiteTracer(10000)` context manager
3. **Configuration-Driven**: Uses YAML configuration to specify which functions to trace
4. **Error Handling**: Logs errors to specified directory if provided
5. **Path Resolution**: Handles relative paths from config file location

## Example Configuration

```yaml
callsite_tracer:
  - file_path: "../../../../tasks/gpt2/unsup/model.py"
    function: main
    traced_functions:
      - fit
      - validate
      - test
```

This will:
1. Find the `main` function in the specified file
2. Wrap any calls to `trainer.fit()`, `trainer.validate()`, or `trainer.test()` with CallsiteTracer
3. Add the necessary import statement

## Integration with OpGuard

The instrumentor is designed to work seamlessly with the existing OpGuard system:

1. **Configuration Sharing**: Uses the same YAML config files as other OpGuard components
2. **CLI Integration**: Available through the main OpGuard CLI
3. **Error Logging**: Integrates with OpGuard's logging system

## Limitations

- Only works with simple function calls (not method calls on objects)
- Requires the function name to match exactly
- Does not handle complex expressions or nested calls
- Assumes the CallsiteTracer constructor takes a `dump_every` parameter

## Future Enhancements

- Support for method calls on objects
- Configurable tracer parameters
- Support for multiple tracer types
- Better error recovery and rollback 