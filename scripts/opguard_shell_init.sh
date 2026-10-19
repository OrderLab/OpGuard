#!/bin/bash
# OpGuard Shell Integration
# Source this file to add opguard shell functions to your current shell
# Usage: source scripts/opguard_shell_init.sh
# Or add to your ~/.bashrc or ~/.zshrc

# OpGuard wrapper function - automatically activates/deactivates monitoring mode
opguard() {
    # Check if this is the 'start' command
    if [ "$1" = "start" ]; then
        # Shift to remove 'start' from arguments
        shift
        # Use process substitution to source the activation commands
        # Pass remaining arguments (if any) to opguard start
        source <(command opguard start "$@")
        return $?
    elif [ "$1" = "stop" ]; then
        # Shift to remove 'stop' from arguments
        shift
        # Use process substitution to source the deactivation commands
        # Pass remaining arguments (if any) to opguard stop
        source <(command opguard stop "$@")
        return $?
    elif [ "$1" = "clear" ]; then
        # Shift to remove 'clear' from arguments
        shift
        # Use process substitution to source the cleanup commands
        # Pass remaining arguments (if any) to opguard clear
        source <(command opguard clear "$@")
        return $?
    elif [ "$1" = "instrument" ]; then
        # Shift to remove 'instrument' from arguments
        shift
        # Use process substitution to source the runtime env exports
        # Pass remaining arguments (if any) to opguard instrument
        source <(command opguard instrument "$@")
        return $?
    elif [ "$1" = "sync" ]; then
        # Shift to remove 'sync' from arguments
        shift
        # For sync, just call opguard normally (no shell commands to source)
        command opguard sync "$@"
        return $?
    else
        # For all other commands, just call opguard normally
        command opguard "$@"
    fi
}

# Export the function
export -f opguard 2>/dev/null || true

echo "[OpGuard] Shell integration loaded. You can now use 'opguard start', 'opguard stop', 'opguard clear', 'opguard instrument', and 'opguard sync' directly."
