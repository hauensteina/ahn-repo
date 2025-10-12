
'''
Convert MacMahon results to AGA format
AHN, Oct 2024
'''

'''
Eample input line:
14	30324 Duan Riannie	2d	xx	xxxx	18	2	18	53	157=	17+/b	11-/w	16+/b

Whitespace delimited, columns:
1. Player placement
2. AGA number
3. LName
4. FName
5. Rank
6. State
7. Club
8. Score
9. Points
10. ScoreX
11. SOS
12. SOSOS

'=' indicates 0.5 points or a draw, depending on the column

Draws are ignored and not submitted for rating.

Example output:
TOURNEY Silicon Valley Go Club Fall Tournament 2024
        start=9/28/2024
        finish=9/28/2004
        rules=AGA
PLAYERS
30324 Duanne, Riannie 2d
25811 Kamalov, Dima	1d
23112 Zhong, Nan	2d
GAMES
23112 30324 b 0 7 // Zhong, Nan vs Duanne, Riannie  black wins, 0 handicap, 7 komi

'''

TOURNEY = '''
TOURNEY Silicon Valley Go Club Fall Tournament 2025
        start=10/11/2025
        finish=10/11/2025
        rules=AGA
'''

from pdb import set_trace as BP
import os, sys
import argparse
import requests

#--------------
def usage():
    name = os.path.basename( __file__)
    msg = f'''
    Name:
      {name}: Convert EGA results export from MacMahon to AGA format
      You need to manually insert the AGA number in the EGA file in column 2 before running this script.
      The script will check with the AGA database.
    Synopsis:
      {name} --egafile <egafile> --outfile <outfile>

    Examples:
        python {name} --egafile results_export.txt --outfile results_aga.txt
        python {name} --fname Liam --lname Liu  # Find AGA number for a player
        python {name} --aganum 20899  # Find player for an AGA number

''' 
    msg += '\n '
    return msg 

#-------------
def main():
    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument( '--egafile')
    parser.add_argument( '--outfile')
    parser.add_argument( '--fname')
    parser.add_argument( '--lname')
    parser.add_argument( '--aganum')
    
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        print( usage())
        sys.exit( 1)
    
    agadict = get_aga_nums()
    
    if args.fname and args.lname:
        find_aganum( args.fname, args.lname, agadict)
        sys.exit( 0)
        
    if args.aganum:
        find_player( args.aganum, agadict)
        sys.exit( 0)

    with open( args.egafile, 'r') as f:
        lines = f.readlines()
        
    players = get_players( lines, agadict)
    games = get_games( lines, players)    
    dump_aga_results( args.outfile, players, games)
    
#-----------------------------------------------
def dump_aga_results( outfile, players, games):    
    with open( outfile, 'w') as f:
        f.write( f'{TOURNEY}')
        f.write( f'PLAYERS\n')
        for p in players:
            (aganum, fname, lname, rank) = players[p]
            f.write( f'{aganum} {lname}, {fname} {rank}\n')
        f.write( f'GAMES\n')
        for g in games:
            (handicap, komi, winner) = games[g]
            (aganum_white, aganum_black) = g.split('_')
            
            f.write( f'{aganum_white} {aganum_black} {winner} {handicap} {komi}\n')

#---------------------------------
def get_games( lines, players):
    games = {} # key = <aganum_white>_<aganum_black>
    for line in lines:  
        try:
            (playernum, aganum, fname, lname, rank, results) = parse_line( line)
        except:
            continue
        for r in results:
            if r.startswith('0'): continue # a bye
            if '=' in r: continue # a draw
                
            mycolor = 'b' if 'b' in r else 'w'
            opponentnum = r.split('/')[0][:-1]
            opponentaganum = players[opponentnum][0]
            handicap = 0
            try:    
                handicap = int(r[-1])
            except:
                pass
            komi = 7
            if handicap > 0:
                komi = 0
            if mycolor == 'w':
                aganum_white = aganum
                aganum_black = opponentaganum
                winner = 'w' if '+' in r else 'b'
            else:
                aganum_black = aganum
                aganum_white = opponentaganum
                winner = 'b' if '+' in r else 'w'

            games[f'{aganum_white}_{aganum_black}'] = (handicap, komi, winner)
    return games

#---------------------------------
def get_players(lines, agadict):
    # Get the players
    players = {} # key = player number starting with 1
    for line in lines:
        try:
            (playernum, aganum, fname, lname, rank, results) = parse_line( line)
        except:
            continue
        if not check_aganum( aganum, fname, lname, agadict):
            sys.exit( 1)
        players[playernum] = (aganum, fname, lname, rank)
    return players            


#---------------------------------
def parse_line( line):
    lien = line.strip()
    parts = line.split()
    if len(parts) < 13:
        return None
    playernum = parts[0]
    aganum = parts[1]
    lname = parts[2]
    lname = lname.replace( '_', ' ')
    fname = parts[3]
    fname = fname.replace( '_', ' ')
    rank = parts[4]
    results = parts[12:]
    return (playernum, aganum, fname, lname, rank, results)

#---------------------------------
def get_aga_nums():
    url = 'https://aga-functions.azurewebsites.net/api/GenerateTDListA'
    r = requests.get( url)
    lines = r.text.split('\n')
    agadict = {}
    for l in lines:
        parts = l.split('\t')
        lname = parts[0].split(',')[0].strip().lower()
        fname = parts[0].split(',')[1].strip().lower()
        aganum = parts[1].strip()
        ttype = parts[2].strip().lower()
        agadict[aganum] = (fname, lname, ttype)
    return agadict    
    
#-----------------------------------------------------    
def find_player( aganum, agadict):
    if aganum in agadict:
        (fname, lname) = agadict[aganum]
        print( f'{fname} {lname}')
    else:
        print( f'AGA number {aganum} not found')
        
#-----------------------------------------------------        
def find_aganum( fname, lname, agadict):
    for aganum in agadict:
        (afname, alname) = agadict[aganum]
        if afname.lower().strip() == fname.lower().strip() and alname.lower().strip() == lname.lower().strip():
            print( f'{fname} {lname} {aganum}')
            return
    print( f'Player {fname} {lname} not found')        
    
#-----------------------------------------------------
def check_aganum( aganum, fname, lname, agadict):
    if aganum in agadict:
        (afname, alname, ttype) = agadict[aganum]
        if afname.lower().strip() == fname.lower().strip() and alname.lower().strip() == lname.lower().strip():
            if ttype == 'non':
                print( f'AGA number {aganum} for {fname} {lname} has expired.')
                return False
            return True
        else:
            print( f'Warning: AGA number {aganum} does not match {fname} {lname}. It points to {afname} {alname}')
            return True
    else:
        print( f'AGA number {aganum} not found')
        return False    
    
main()
