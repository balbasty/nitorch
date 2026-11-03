"""Implementation of the command-line entry point for nitorch's tools.

Subcommands should register themselves by doing:
```python
>> from nitorch.tools.cli import commands
>> commands['commandname'] = command_fn
```
The registered function should be a parser which takes the list of
command-line arguments (without the command name) as input.

Exit status: a command function reports a failure (a `ParseError` or any
other exception it caught and printed as `[ERROR] ...`) by returning a
non-zero integer (by convention, 1); it must not call `sys.exit` itself, so
that it can be called programmatically. `cli` turns that into the return
value of the `nitorch` executable (the console script and
`python -m nitorch` pass it to `sys.exit`). Any other return value
(including `None`) means success.
"""
import sys


# Subcommands should registers themselves in this dictionary
commands = dict()

_help = r"""[nitorch] Command-line tools

usage:
    nitorch <COMMAND>
    nitorch help <COMMAND>
        
"""


def help():
    """Help for the generic 'nitorch' command.
    List all registered commands.
    """

    commandnames = sorted(commands.keys())
    length = max(len(name) for name in commandnames)
    nb_col = 76//(length+1)
    commandlist = "    "
    for d, name in enumerate(commandnames):
        commandlist += ('{:<' + str(length) + 's} ').format(name)
        if (d+1) % nb_col == 0:
            commandlist += '\n    '
    commandlist += '\n'

    return _help + commandlist


def cli(args=None):
    """Generic parser for nitorch commands.
    This function calls the appropriate subcommand if it is registered.
    """

    args = args or sys.argv[1:]  # remove command name

    if not args:
        print(help())
        return
    tag, *args = args
    if tag == 'help':
        if args:
            if args[0] in commands.keys():
                return _status(commands[args[0]](['-h']))
            else:
                print(help())
                print(f'[ERROR] Unknown command "{args[0]}"', file=sys.stderr)
                return 1
        else:
            print(help())
            return
    if tag in ('-h', '--help'):
        print(help())
        return
    if tag not in commands.keys():
        print(help())
        print(f'[ERROR] Unknown command "{tag}"', file=sys.stderr)
        return 1
    return _status(commands[tag](args))


def _status(rc):
    """Convert the return value of a command into an exit status.

    Only a non-zero integer (not a bool) signals a failure; commands may
    return arbitrary objects on success, which must not reach `sys.exit`
    (it would print them and exit with status 1).
    """
    if isinstance(rc, int) and not isinstance(rc, bool) and rc != 0:
        return rc
    return 0
