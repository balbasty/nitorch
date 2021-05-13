"""Implementation of the command-line entry point for nitorch's tools.

Subcommands should register themselves by doing:
```python
>> from nitorch.tools.cli import commands
>> commands['commandname'] = command_fn
```
The registered function should be a parser which takes the list of
command-line arguments (without the command name) as input.
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
                return commands[args[0]](['-h'])
            else:
                print(help)
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
    return commands[tag](args)
