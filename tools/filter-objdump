from signal import signal, SIGPIPE, SIG_DFL
import sys

def _replace_all(seq1, seq2, s):
    while seq1 in s:
        s = s.replace(seq1, seq2)
    return s

if __name__ == '__main__':
    signal(SIGPIPE, SIG_DFL)
    for line in sys.stdin:
        if not line.startswith(' '):
            continue
        elems = line.replace('\n', '').split('\t')

        if len(elems) != 3:
            continue

        elems = elems[1:]

        opcode = elems[0].replace(' ', '')
        disasm = _replace_all('  ', ' ', elems[1])

        if '#' in disasm:
            disasm = disasm[:disasm.find('#')]
        print('{}|{}'.format(opcode, disasm))