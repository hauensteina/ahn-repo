import os
import argparse
from pdb import set_trace as BP
import shutil


#-------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Renumber flashcard files')
    parser.add_argument('--folder', type=str, help='folder containing the SVG files', required=True)
    parser.add_argument('--offset', type=int, help='Number of first card', default = 10,)
    parser.add_argument('--delta', type=int, help='Difference between card numbers', default = 10)
    args = parser.parse_args()
    
    renum_files(args.folder, args.offset, args.delta)
  
#-------------------------------------------------------------  
def renum_files(folder, offset, delta):
    """ 
    Filenames look like folder/0003_f_2.svg, where the first number is the card number.
    We want to sort the files by card number, and rename them such that all files for the lowest card
    get the offset as the new number, and subsequent card files get the offset + delta, etc.
    """
    files = os.listdir(folder)
    files = [f for f in files if f.endswith('.svg')]
    if not os.path.exists(folder + '/old'):
        os.mkdir(folder + '/old')

    # Back up originals        
    for f in files:
        os.rename(folder + '/' + f, folder + '/old/' + f)
    files = os.listdir(folder + '/old')
    cardnums = sorted(list(set([int(f.split('_')[0]) for f in files])))
        
    # Rename files    
    for i,cardnum in enumerate(cardnums):
        new_num = offset + i*delta
        cardfiles = [f for f in files if f.startswith(f"{cardnum:04d}_")]
        for f in cardfiles:
            newname = f"{new_num:04d}" + f[4:]
            # Copy the file to the new name
            shutil.copyfile(folder + '/old/' + f, folder + '/' + newname)
            print(f"Renamed {f} to {newname}")

main()
