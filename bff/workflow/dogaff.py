import os
from pathlib import Path
import sys

def main(fn_pdb, resname):

    # Create the parent directory
    gaff_dir = Path('./gaff2').resolve()
    print(gaff_dir)
    gaff_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(gaff_dir)

    os.system(f'antechamber -i {fn_pdb} -fi pdb -o {resname}.mol2 -fo mol2 -c bcc -s 2')
    os.system(f'parmchk2 -i {resname}.mol2 -f mol2 -o {resname}.frcmod')

    # Create the tleap file
    with open('tleap.in', 'w') as f:
        print('source leaprc.ff99SB', file=f)
        print('source leaprc.gaff', file=f)
        print(f'{resname} = loadmol2 {resname}.mol2', file=f)
        print(f'check {resname}', file=f)
        print(f'loadamberparams {resname}.frcmod', file=f)
        print(f'saveoff {resname} {resname}.lib', file=f)
        print(f'saveamberparm {resname} {resname}.prmtop {resname}.inpcrd', file=f)
        print('quit', file=f)


    os.system('tleap -f tleap.in')

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
