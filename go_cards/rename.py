import os
import argparse
from pdb import set_trace as BP
import shutil


#-------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Renumber flashcard files')
    parser.add_argument('--folder', type=str, help='folder containing the SVG files', required=True)
    parser.add_argument('oldnum', type=int, help='Card to move', default = 10,)
    parser.add_argument('newnum', type=int, help='New card number', default = 10)
    args = parser.parse_args()
    
    rename_card(args.folder, args.oldnum, args.newnum)
  
#-------------------------------------------------------------  
def rename_card(folder, oldnum, newnum):
    """ 
    Filenames look like folder/0003_f_2.svg, where the first number is the card number.
    """
    files = os.listdir(folder)
    files = [f for f in files if f.endswith('.svg')]
    oldfiles = [f for f in files if f.startswith(f"{oldnum:04d}_")]
    newfiles = [f for f in files if f.startswith(f"{newnum:04d}_")]
    if newfiles:
        print(f"Error: {newnum:04d} already exists")
        return
    
    newfiles = [f"{newnum:04d}" + f[4:] for f in oldfiles]
    for old,new in zip(oldfiles,newfiles):
        os.rename(folder + '/' + old, folder + '/' + new)
        print(f"Renamed {old} to {new}")

main()
