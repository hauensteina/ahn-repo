
'''
AHAUX invoice PDF generator.
AHN, Jan 2022
'''

from pdb import set_trace as BP
from datetime import datetime
from fpdf import FPDF

rows = [
    { 'DESCRIPTION': 'Kickoff', 'DATE':'2023-01-02', 'RATE':'USD 1400', 'QUANTITY':0.5 }
    ,{ 'DESCRIPTION': 'Work', 'DATE':'2023-01-03', 'RATE':'USD 1400', 'QUANTITY':1.0 }
    ,{ 'DESCRIPTION': 'Finish', 'DATE':'2023-01-04', 'RATE':'USD 1400', 'QUANTITY':0.5 }
]
rows = rows * 15

NCOLS = 4
WHITE = (255,255,255)
GRAY = (220,220,220)
DARK = (90,90,90)
BLACK = (0,0,0)
RED = (0xE0,0x0A,0x1A)
FONTSIZE = 8

class PDF(FPDF):
    def header(self):
        if self.page_no() == 1: self.front_header()
        self.table_header()
     
    def front_header(self, customer='D ONE\nSihlfeldstrasse 58\n8003 Zurich\nSwitzerland', invoice_no=42, terms='on receipt'):
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
        self.multi_cell( 0, spacing, customer, align='L', new_x='RIGHT', new_y='NEXT', border=0)
        
        self.ln(5)
        self.set_text_color(*BLACK)

    def table_header(self):
        self.set_font('Helvetica', size=FONTSIZE)
        line_height = self.font_size * 2.0
        col_width = self.epw / NCOLS       
        self.set_font(style='B')
        self.set_fill_color( *GRAY)
        for col in rows[0]:
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
        
def main():
    pdf = PDF()
    pdf.set_margins( left=15, top=10, right=15)
    pdf.add_page()
    pdf.set_font('Helvetica', size=FONTSIZE)
    pdf.set_auto_page_break( True, margin=20)
    line_height = pdf.font_size * 2.0
    col_width = pdf.epw / NCOLS
    for rnum,row in enumerate(rows):
        pdf.set_font(style='')
        pdf.set_fill_color( 255,255,255)
        for col in row:
            align = 'R'
            if col == 'DATE': align = 'C'
            elif col == 'DESCRIPTION': align = 'L'
            pdf.multi_cell( col_width, line_height, ' ' + str(row[col]) + ' ', border=1,
                           new_x='RIGHT', new_y='TOP', max_line_height=pdf.font_size, fill=True, align=align)
        pdf.ln( line_height)

    pdf.output('invoice.pdf')


main()
