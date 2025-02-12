from pdb import set_trace as BP
import os
import subprocess
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import xml.etree.ElementTree as ET
import argparse

PAGE_WIDTH, PAGE_HEIGHT = letter  # Letter size: 8.5" x 11"
SCALE_FACTOR = 0.85

#-------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Generate flashcards from svg files')
    parser.add_argument('--folder', type=str, help='folder containing the SVG files', required=True)
    parser.add_argument('--title', type=str, help='title of the flashcards', required=False)
    args = parser.parse_args()
    
    if not args.title:
        args.title = args.folder
    
    generate_blank_svg('blank.svg')
    svg_files = [f'{args.folder}/{f}' for f in os.listdir(args.folder) if f.endswith('.svg')]
    output_pdf = f'{args.folder}/{args.folder}_flashcards.pdf'
    canv = canvas.Canvas(output_pdf, pagesize=letter)
    
    print_title(canv, args.title)   
    print_title(canv, "")
    
    cards = sorted(list(set([ os.path.split(x)[-1].split('_')[0] for x in svg_files ])))
    
    # Go through cards in groups of 4
    for i in range(0, len(cards), 4):
        print(f"Processing cards {i+1} to {i+4}")
        
        # Four cards make a page
        if i+4 > len(cards):
            cards_group = cards[i:]
        else:
            cards_group = cards[i:i+4] 
        svg_files_group = [f for f in svg_files if f'{os.path.split(f)[-1].split("_")[0]}' in cards_group]      
        
        generate_flashcards(canv, svg_files_group, i, len(cards))

    canv.save()
    os.remove('blank.svg')
    print()
    print("Ignore any warnings")
    print(f"Flashcards generated in '{output_pdf}'")

#------------------------------------------------------------------------
def print_title(canv, title):
    scale_canvas(canv)
    draw_cutting_marks(canv)
    card_height = PAGE_HEIGHT / 2
    card_width = PAGE_WIDTH / 2
    font = "Helvetica"
    size = 16
    textwidth = canv.stringWidth(title, font, size)
    canv.setFont(font, size)
    canv.drawString( card_width/2 - textwidth/2, 
                    card_height/2, 
                    title)
    canv.showPage()

#-----------------------------------------------------------------------------------------
def generate_flashcards(canv, svg_files, numbering_offset, total_cards):
    """
        Draw two pages on a canvas.
        Four cards fit on a duplex sheet.
        The first page has the front side of the cards.
        The second page has the back side of the cards.
        Each side of a card has one or two diagrams.    
    """

    get_first_svg = lambda lst: lst[0] if lst else 'blank.svg'
            
    # 'double_44/0004_f_2.svg' -> '0004'
    cards = sorted(list(set([ os.path.split(x)[-1].split('_')[0] for x in svg_files ])))
    if len(cards) % 2 != 0:
        cards.append('xxxx')
        
    # Arrange front side (first page)
    scale_canvas(canv)
    draw_cutting_marks(canv)
    for i, card in enumerate(cards):
        front_dia_1 = get_first_svg([ x for x in svg_files if x.endswith(f"{card}_f_1.svg") ]) # top
        front_dia_2 = get_first_svg([ x for x in svg_files if x.endswith(f"{card}_f_2.svg") ]) # bot
        row = i // 2
        col = i % 2

        draw_svg_on_canvas(canv, front_dia_1, 2 * row, col)
        draw_svg_on_canvas(canv, front_dia_2, 2 * row + 1, col, f'{i+numbering_offset+1}/{total_cards}')
        
    canv.showPage()

    # Arrange back side (second page)
    scale_canvas(canv)
    draw_cutting_marks(canv)
    for i, card in enumerate(cards):
        try:
            backcard = cards[i+1] if i%2 == 0 else cards[i-1]
        except:
            backcard = 'xxxx'
        back_dia_1 = get_first_svg([ x for x in svg_files if x.endswith(f"{backcard}_b_1.svg") ]) # top
        back_dia_2 = get_first_svg([ x for x in svg_files if x.endswith(f"{backcard}_b_2.svg") ]) # bot

        row = i // 2
        col = i % 2

        draw_svg_on_canvas(canv, back_dia_1, 2 * row, col)
        draw_svg_on_canvas(canv, back_dia_2, 2 * row + 1, col)

    canv.showPage()
        
#-------------------------------------------------------------
def draw_rect(canvas, rect):
    """ Useful for layout debugging """
    x, y, width, height = rect
    canvas.setLineWidth(1)
    canvas.setStrokeColorRGB(0, 0, 0)
    canvas.setFillColorRGB(1, 1, 1)
    canvas.rect(x, y, width, height)

#-------------------------------------------------------------        
def draw_line(canvas, x1, y1, x2, y2):
    canvas.line(x1, y1, x2, y2)        

#----------------------------------------------------------------------------
def convert_svg_to_png_inkscape(svg_file, png_file, width, height):
    """Convert an SVG to PNG with a specific width and height using Inkscape."""
    command = [
        "inkscape", svg_file, 
        "--export-type=png", 
        "--export-filename=" + png_file, 
        f"--export-width={width}", 
        f"--export-height={height}",
        "--export-dpi=600",
        "--without-gui"
    ]
    subprocess.run(command, check=True)
    
#------------------------------------------------------------------------
def draw_svg_on_canvas(canvas, svg_file, row, col, footer = ''):
    """
    Draw an SVG file on the ReportLab canvas using svglib.
    row in (0,1,2,3) 
    col in (0,1)
    y goes from bottom to top
    """
    if not os.path.exists(svg_file):
        return
    svg_width, svg_height = get_svg_dimensions(svg_file)
    card_height = PAGE_HEIGHT / 2
    middle = PAGE_HEIGHT / 2
    card_width = PAGE_WIDTH / 2
    hgap = PAGE_WIDTH * 0.05
    vgap = card_height * 0.05
    botmarg = card_height * 0.05
    topmarg = card_height * 0.05
    diagram_width = (PAGE_WIDTH - 2 * hgap) / 2
    diagram_height = (card_height - botmarg - topmarg - vgap) / 2
    x = hgap / 2 + col * (diagram_width + hgap)
    if row == 0: y = PAGE_HEIGHT - topmarg - diagram_height 
    elif row == 1: y = PAGE_HEIGHT - topmarg - vgap - 2 * diagram_height
    elif row == 2: y = middle - topmarg - diagram_height
    else: y = middle - topmarg - vgap - 2 * diagram_height 
                 
    #rect = (x, y, diagram_width, diagram_height)
    #draw_rect(canvas, rect)
    # Scale and center the SVG on the rectangle
    scale_x = diagram_width / svg_width
    scale_y = diagram_height / svg_height
    scale = min(scale_x, scale_y)
    scale = scale_y
    
    dx = (diagram_width - svg_width*scale) / 2
    dy = (diagram_height - svg_height*scale) / 2
    
    # Convert to png
    png_file = svg_file.replace('.svg', '.png')
    
    upsample_factor = 3 # reduce pixelation
    convert_svg_to_png_inkscape(svg_file, png_file, width=int(upsample_factor*svg_width*scale), height=int(upsample_factor*svg_height*scale))
    canvas.drawImage(png_file, x + dx, y + dy, width=svg_width*scale, height=svg_height*scale)
    os.remove(png_file)
    
    if footer:
        # Draw footer
        font = "Helvetica"
        size = 10
        textwidth = canvas.stringWidth(footer, font, size)
        canvas.setFont(font, size)
        canvas.drawString(card_width * col + card_width/2 - textwidth/2, 
                            (3 - row) / 2 * card_height + botmarg * 0.4, 
                            footer)
    
#-------------------------------------------------------------        
def get_svg_dimensions(svg_file):
    """Extracts width and height from the <svg> tag in an SVG file."""
    tree = ET.parse(svg_file)
    root = tree.getroot()
    width = root.attrib.get("width")
    height = root.attrib.get("height")

    return float(width), float(height)

#-------------------------------------------------------------        
def generate_blank_svg(fname): 
    with open(fname, 'w') as f:
        f.write('<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="100" height="100"></svg>')
  
#---------------------------------------------------------------------------
def scale_canvas(c):
    """Scale and center canvas"""
    scaled_width = PAGE_WIDTH * SCALE_FACTOR
    scaled_height = PAGE_HEIGHT * SCALE_FACTOR
    translate_x = (PAGE_WIDTH - scaled_width) / 2
    translate_y = (PAGE_HEIGHT - scaled_height) / 2
    c.translate(translate_x, translate_y)  # Move origin to center
    c.scale(SCALE_FACTOR, SCALE_FACTOR)    # Scale content

#-------------------------------------------------------------
def draw_cross(c,x,y):
    MARKLEN=5
    c.line(x - MARKLEN, y, x + MARKLEN, y)
    c.line(x, y - MARKLEN, x, y + MARKLEN)

#-------------------------------------------------------------
def draw_cutting_marks(c):
    c.setLineWidth(0.5)
    hmarg = (PAGE_WIDTH - PAGE_WIDTH * SCALE_FACTOR) / 2 - 10
    vmarg = (PAGE_HEIGHT - PAGE_HEIGHT * SCALE_FACTOR) / 2 - 10
    
    # bottom left crosses
    draw_cross(c, -hmarg,0)
    draw_cross(c, 0,-vmarg)
    draw_cross(c, 0,0)

    # bottom right crosses
    draw_cross(c, PAGE_WIDTH + hmarg,0)
    draw_cross(c, PAGE_WIDTH, -vmarg)
    draw_cross(c, PAGE_WIDTH, 0)

    # top left crosses
    draw_cross(c, -hmarg, PAGE_HEIGHT)
    draw_cross(c, 0, PAGE_HEIGHT + vmarg)
    draw_cross(c, 0, PAGE_HEIGHT)

    # top right crosses
    draw_cross(c, PAGE_WIDTH + hmarg, PAGE_HEIGHT)
    draw_cross(c, PAGE_WIDTH, PAGE_HEIGHT + vmarg)
    draw_cross(c, PAGE_WIDTH, PAGE_HEIGHT)
    
    # top middle cross
    draw_cross(c, PAGE_WIDTH/2, PAGE_HEIGHT + vmarg)
    draw_cross(c, PAGE_WIDTH/2, PAGE_HEIGHT)
    
    # bottom middle cross
    draw_cross(c, PAGE_WIDTH/2, -vmarg)
    draw_cross(c, PAGE_WIDTH/2, 0)
    
    # left middle cross
    draw_cross(c, -hmarg, PAGE_HEIGHT/2)
    draw_cross(c, 0, PAGE_HEIGHT/2)
    
    # right middle cross
    draw_cross(c, PAGE_WIDTH + hmarg, PAGE_HEIGHT/2)
    draw_cross(c, PAGE_WIDTH, PAGE_HEIGHT/2)  
    
    # center cross
    draw_cross(c, PAGE_WIDTH/2, PAGE_HEIGHT/2)  
        
main()