import os
import random
import re
from gmgcode.groups.canons import canonicals
from gmgcode.pcs.pcs_object import PCSet

def clear():
    print('\n' * 50)

def print_state(pcset: PCSet):
    print('=== Pitch Class Set Explorer ===')
    print(f"Binary:  {''.join(map(str, pcset.binary))[::-1]}")
    print(f'Decimal: {pcset.decimal}')
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

def run_game():
    print('=== Pitch Class Set Explorer ===')
    n = int(input('Equal temperament (e.g., 12): '))
    k = int(input('Cardinality (number of notes): '))

    canon = list(canonicals(n, k))
    if not canon:
        print('No canonical sets found.')
        return

    print(f'{len(canon)} canonical pitch-class sets found.')
    print('Choose starting mode:')
    print(' 1. Random')
    print(' 2. Manually enter binary string')
    print(' 3. Select from list')
    print(' q. Quit')

    choice = input('Your choice: ').strip()
    if choice == 'q':
        return

    if choice == '1':
        pc = PCSet(random.choice(canon))

    elif choice == '2':
        bits = input(f'Enter binary string ({n} bits): ').strip()
        try:
            pc = PCSet(int(bits, 2))
        except ValueError:
            print('Invalid input.')
            return

    elif choice == '3':
        for i, val in enumerate(canon):
            pcs = PCSet(val)
            print(f'{i:3}: {pcs.binary} → {pcs.pcs}')
        idx = int(input('Select index: '))
        pc = PCSet(canon[idx])

    else:
        print('Invalid choice.')
        return

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
            pc = pc << shift

        elif m := re.match(r'^sr(\d+)$', cmd):
            axis = int(m.group(1)) % n
            pc = PCSet(pc.reflect(axis))

        elif m := re.match(r'^u(\d+)$', cmd):
            idx = int(m.group(1))
            if idx < len(pc):
                p = pc[idx]
                new = (p + 1) % n
                if new not in pc:
                    pcs = list(pc.pcs)
                    pcs[idx] = new
                    pc = PCSet(pcs)

        elif m := re.match(r'^d(\d+)$', cmd):
            idx = int(m.group(1))
            if idx < len(pc):
                p = pc[idx]
                new = (p - 1) % n
                if new not in pc:
                    pcs = list(pc.pcs)
                    pcs[idx] = new
                    pc = PCSet(pcs)

        else:
            input('Invalid command. Press Enter to continue.')



if __name__ == '__main__':
    run_game()
    
    