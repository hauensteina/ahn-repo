
'''
AHAUX invoice PDF generator.
AHN, Jan 2022
'''

from pdb import set_trace as BP
from datetime import datetime
import os
import argparse
from fpdf import FPDF # This is fpdf2, pip install fpdf2, not fpdf 
import csv

CUSTOMER_ADDRESS = {
    'D1': 'D ONE\nSihlfeldstrasse 58\n8003 Zurich\nSwitzerland'
    ,'PROTXX': 'John Ralston\nPROTXX, INC.\n369 La Cuesta Drive\nPortola Valley, CA 94028'
    ,'Neursantys': 'Neursantys, INC.\n659 Oak Grove Ave\nMenlo Park, CA 94025'
}

COL_WIDTH = {
    'DESCRIPTION':100
    ,'DATE':20
    ,'RATE':20
    ,'QUANTITY':20
    ,'AMOUNT':20
}

WHITE = (255,255,255)
GRAY = (220,220,220)
DARK = (90,90,90)
BLACK = (0,0,0)
RED = (0xE0,0x0A,0x1A)
FONTSIZE = 8

class PDF(FPDF):
    def __init__( self, rows, customer_addr, invoice_no, terms):
        FPDF.__init__(self)
        self.rows = rows
        self.customer_addr = customer_addr
        self.invoice_no = invoice_no
        self.terms = terms

    def header(self):
        if self.page_no() == 1: self.front_header( self.customer_addr, self.invoice_no, self.terms)
        self.table_header()
     
    def front_header(self, customer_addr, invoice_no, terms):
        self.set_font('Helvetica', size=30)
        self.set_text_color(*DARK)
        # Heading
        self.cell( 100, 20, 'AHAUX LLC', align='L', border=0)
        self.cell( 0, 20, 'Invoice', align='R', border=0)
        self.ln()
        # Address
        spacing = 3.7
        self.set_font('Helvetica', size=FONTSIZE)
        self.set_text_color(*BLACK)
        txt = f'3040 Boyter Pl Unit 105\nSanta Clara, CA 95051\nUSA\nPhone: +1-415-706-0740\nEmail: hauensteina@gmail.com'
        self.multi_cell( 140, spacing, txt, align='L', new_x='RIGHT', new_y='TOP', border=0)
        # Middle
        txt = f'DATE:\nINVOICE NO.\nTERMS:'
        self.multi_cell( 20, spacing, txt, align='L', new_x='RIGHT', new_y='TOP', border=0)
        # Right
        self.set_fill_color( *GRAY)
        date = datetime.today().isoformat().split('T')[0]
        txt = f'{date}\n[{invoice_no:05d}]\n{terms}'
        self.set_text_color(*RED)
        self.multi_cell( 0, spacing, txt, align='C', new_x='RIGHT', new_y='NEXT', fill=True, border=0)
        self.ln(10)
        # Bill To
        self.set_text_color(*BLACK)
        self.set_fill_color( *GRAY)
        self.cell( 50, spacing, 'BILL TO:', align='L', new_x='LEFT', new_y='NEXT', fill=True, border=0)
        self.set_text_color(*RED)
        self.multi_cell( 0, spacing, customer_addr, align='L', new_x='RIGHT', new_y='NEXT', border=0)
        
        self.ln(5)
        self.set_text_color(*BLACK)

    def table_header(self):
        self.set_font('Helvetica', size=FONTSIZE)
        line_height = self.font_size * 2.0
        self.set_font(style='B')
        self.set_fill_color( *GRAY)
        for col in self.rows[0]:
            col_width = COL_WIDTH[col]
            self.multi_cell( col_width, line_height, col, border=1,
                            new_x='RIGHT', new_y='TOP', max_line_height=self.font_size, fill=True, align='C')
        self.ln( line_height)

    def footer(self):
        if self.page_no() == 1:
            self.front_page_footer()
            return
        # Distance from bottom
        self.set_y(-17)
        # Select Arial italic 8
        self.set_font( 'Helvetica', '', 8)
        # Print current and total page numbers
        self.cell( 0, 10, 'Page %s' % self.page_no() + '/{nb}',align='C')

    def front_page_footer(self):
        ''' Bank Info '''
        txt = f'\n\n\n\nAccount: 325108158898  Routing Nr: 121000358  SWIFT/BIC: BOFAUS3N\nBank of America, NA\n222 Broadway\nNew York, New York 10038'
        self.set_y(-35)
        self.multi_cell( 0, 3.7, txt, align='L', new_x='RIGHT', new_y='NEXT', border=0)
        self.ln(10)
  
#--------------
def usage():
    name = os.path.basename( __file__)
    msg = f'''
    Name:
      {name}: Generate an AHAUX invoice from time sheet csv

    Synopsis:
      {name} --timesheet <fname>.csv --customer [D1|PROTXX|Neursantys] [--terms <str>] [--invoice_no <int>]

    Description:
      Generate an invoice from a csv. All upper case columns show in the output PDF.
      Column width is set in the COL_WIDTH dict in the source code.

    Example:
      python {name} --timesheet timesheet.csv --customer Neursantys --invoice_no 2

    Default for terms is 'on receipt'.
    Default for invoce_no if the largest invoce_no found in the csv.
    Output goes to files <fname>.pdf .

''' 
    msg += '\n '
    return msg 

#-------------
def main():
    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument( '--timesheet', required=True)
    parser.add_argument( '--customer', required=True)
    parser.add_argument( '--invoice_no', type=int)
    parser.add_argument( '--terms', default='on receipt')
    args = parser.parse_args()
    with open(args.timesheet) as inf: csvstr = inf.read()
    rows, colnames = csv2dict( csvstr)

    if int(args.invoice_no) > 0:
        invoice_no = args.invoice_no
    else:
        invoice_no = max( [ int(r['invoice_no']) for r in rows ])
    
    rows = [ r for r in rows if r['invoice_no'] == invoice_no ]
    total_amount = 0.0
    total_quantity = 0.0
    newrows = []

    for r in rows: 
        currency, rate = r['RATE'].split()
        quantity = float(r["QUANTITY"])
        quantity = float(f'{quantity:.2f}')
        amount = float(rate) * quantity
        total_quantity += quantity
        amount = int(amount + 0.5)
        total_amount += amount
        r['AMOUNT'] = f'{amount}'
        r['QUANTITY'] = f'{quantity:.2f}'
        
        # Only keep uppercase columns
        r = { k:v for k,v in r.items() if k == k.upper() }
        newrows.append(r)

    rows = newrows    
    customer_addr = CUSTOMER_ADDRESS[args.customer]
    outfname = os.path.splitext( args.timesheet)[0] + '.pdf'
    
    run( rows, customer_addr, invoice_no, args.terms, total_quantity, total_amount, outfname)

#-------------------------------------------------------------
def run( rows, customer_addr, invoice_no, terms, total_quantity, total_amount, outfname):
    pdf = PDF(rows, customer_addr, invoice_no, terms)
    pdf.set_margins( left=15, top=10, right=15)
    pdf.add_page()
    pdf.set_font('Helvetica', size=FONTSIZE)
    pdf.set_auto_page_break( True, margin=20)
    line_height = pdf.font_size * 2.0

    # Itemized rows
    for rnum,row in enumerate(rows):
        pdf.set_font(style='')
        pdf.set_fill_color( 255,255,255)
        for col in row:
            col_width = COL_WIDTH[col]
            align = 'R'
            if col == 'DATE': align = 'C'
            elif col == 'DESCRIPTION': align = 'L'
            pdf.multi_cell( col_width, line_height, str(row[col]), border=1,
                            new_x='RIGHT', new_y='TOP', max_line_height=pdf.font_size, fill=True, align=align)
        pdf.ln( line_height)

    # The total row
    currency = rows[0]['RATE'].split()[0]
    pdf.set_font(style='B')
    pdf.multi_cell( COL_WIDTH['DESCRIPTION'] + COL_WIDTH['DATE']+COL_WIDTH['RATE'], line_height, 'TOTAL', border=1, new_x='RIGHT', new_y='TOP', 
                    max_line_height=pdf.font_size, fill=True, align='C')
    pdf.multi_cell( COL_WIDTH['QUANTITY'], line_height, f'{total_quantity:.2f}', border=1, new_x='RIGHT', new_y='TOP', 
                    max_line_height=pdf.font_size, fill=True, align='R')
    pdf.multi_cell( COL_WIDTH['AMOUNT'], line_height, f'{currency} {total_amount:.0f}', border=1, new_x='RIGHT', new_y='TOP', 
                    max_line_height=pdf.font_size, fill=True, align='R')
    pdf.set_font(style='')

    pdf.output(outfname)

# Transform csv format to a list of dicts
#-------------------------------------------
def csv2dict( csvstr):

    def split_string(s):
        reader = csv.reader([s])
        return next(reader)
    
    lines = csvstr.split('\n')
    colnames = []
    res = []
    for idx, line in enumerate( lines):
        line = line.strip()
        if len(line) == 0: continue
        if line[0] == '#': continue
        words = split_string(line)
        words = [w.strip() for w in words]
        words = [w.strip('"') for w in words]
        words = [w.strip("'") for w in words]
        if not colnames:
            colnames = words
            continue
        ddict = { col:number(words[idx]) for idx,col in enumerate(colnames) }
        res.append(ddict)
    return res, colnames

# Convert a string to a float, if it is a number
#--------------------------------------------------
def number( tstr):
    try:
        res = float( tstr)
        return res
    except ValueError:
        return tstr

main()
