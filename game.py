# game.py

import re
from .pcs_object import PCSet

def clear():
    print('\n' * 50)

def print_state(pcset: PCSet):
    print('=== Pitch Class Set Explorer ===')
    print(f"Binary:  {''.join(map(str, pcset.binary))[::-1]}")
    print(f'Decimal: {pcset.mask}')
    print(f'PCS:     {list(pcset.pcs)}')
    print('-' * 30)

def list_commands():
    print('Commands:')
    print('  rN   = rotate CCW by N semitones')
    print('  srN  = reflect about axis through pitch N')
    print('  uN   = move the Nth note up by 1 semitone')
    print('  dN   = move the Nth note down by 1 semitone')
    print('  q    = quit')
    print()

def start_with(pc: PCSet):
    n = pc._ET

    while True:
        clear()
        print_state(pc)
        list_commands()
        cmd = input('>> ').strip().lower()

        if cmd == 'q':
            print('Goodbye!')
            break

        elif m := re.match(r'^r(\d+)$', cmd):
            shift = int(m.group(1)) % n
            pc = PCSet(pc << shift, n)

        elif m := re.match(r'^sr(\d+)$', cmd):
            axis = int(m.group(1)) % n
            pc = PCSet(pc.transform.reflect(axis), n)

        elif m := re.match(r'^u(\d+)$', cmd):
            idx = int(m.group(1))
            if idx < len(pc):
                p = pc[idx]
                new = (p + 1) % n
                if new not in pc:
                    pcs = list(pc.pcs)
                    pcs[idx] = new
                    pc = PCSet(pcs, n)

        elif m := re.match(r'^d(\d+)$', cmd):
            idx = int(m.group(1))
            if idx < len(pc):
                p = pc[idx]
                new = (p - 1) % n
                if new not in pc:
                    pcs = list(pc.pcs)
                    pcs[idx] = new
                    pc = PCSet(pcs, n)

        else:
            input('Invalid command. Press Enter to continue.')