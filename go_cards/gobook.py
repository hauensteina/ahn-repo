from playwright.sync_api import sync_playwright
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import xml.etree.ElementTree as ET
from PIL import Image
import svgwrite
import textwrap
import json

import base64
import os
import argparse
from pdb import set_trace as BP

LETTER_WIDTH, LETTER_HEIGHT = letter  # Letter size: 8.5" x 11"
PAGE_HEIGHT = LETTER_HEIGHT 
PAGE_WIDTH = LETTER_WIDTH * 0.80 # Cards should be narrower than a quarter letter size
SCALE_FACTOR = 0.85 # Cards should be smaller than a quarter letter size

#-------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Generate flashcards from svg files')
    parser.add_argument('--config', type=str, help='Json file with the chapters', required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)
    title = config['title']
    chapters = config['chapters']    
    
    book_folder = args.config.split('/')[-1].split('.')[0]
    os.makedirs(book_folder, exist_ok=True)
    title_svg = f'{book_folder}/title.svg'
    create_title_svg(title, title_svg, LETTER_WIDTH, LETTER_HEIGHT, 30)
    blank_svg = f'{book_folder}/blank.svg'
    generate_blank_svg(blank_svg)
    
    output_pdf = f'{book_folder}/{book_folder}_book.pdf'
    canv = canvas.Canvas(output_pdf, pagesize=letter)

    for chapter in chapters[:1]:
        BP()
        generate_chapter(chapter['folder'], chapter['title'], canv)

    canv.save()

#--------------------------------------------------------------
def generate_chapter( chapter_folder, title, canv):
    title_svg = f'{chapter_folder}/title.svg'
    create_title_svg(title, title_svg, LETTER_WIDTH, LETTER_HEIGHT, 30)
    blank_svg = f'{chapter_folder}/blank.svg'
    generate_blank_svg(blank_svg)

    svg_files = [f'{chapter_folder}/{f}' for f in os.listdir(chapter_folder) if f.endswith('.svg')] 
    png_files = [ f'{x.replace(".svg", ".png")}' for x in svg_files ]
    
    # Convert SVG to PNG
    batch_convert_svgs(chapter_folder, chapter_folder, 2.0)
        
    # output_pdf = f'{args.folder}/{args.folder}_flashcards.pdf'
    
    # #print_title(canv, args.title)   
    # #print_title(canv, "")
    
    cards = sorted(list(set([ os.path.split(x)[-1].split('_')[0] for x in svg_files ])))
    cards = [x for x in cards if not 'blank' in x]

    # Go through cards in groups of 4
    for i in range(0, len(cards), 4):
        print(f"Processing cards {i+1} to {i+4}")
        
        # Four cards make a page
        if i+4 > len(cards):
            cards_group = cards[i:]
        else:
            cards_group = cards[i:i+4] 
        png_files_group = [f for f in png_files if f'{os.path.split(f)[-1].split("_")[0]}' in cards_group]      
        svg_files_group = [f for f in svg_files if f'{os.path.split(f)[-1].split("_")[0]}' in cards_group]
        captions_group = { f.replace('.svg', '.png'):get_caption(open(f).read()) for f in svg_files_group }
        
        generate_flashcards(canv, png_files_group, captions_group, i, len(cards), f'{chapter_folder}/blank.png')

    os.remove(blank_svg)
    for f in png_files: os.remove(f)
    print()
    print(f"Generated chapter {title} in {chapter_folder}")

#----------------------------------------------------------------------------------
def create_title_svg(title, output_path, width, height, font_size, max_chars=30):
    # Create an SVG drawing
    dwg = svgwrite.Drawing(output_path, size=(f"{width}", f"{height}"))
    wrapped_lines = textwrap.wrap(title, width=max_chars)
    line_height = font_size * 1.3  
    total_text_height = len(wrapped_lines) * line_height
    start_y = (height - total_text_height) / 2 + font_size 
    
    # Add each line to the SVG
    for i, line in enumerate(wrapped_lines):
        y_position = start_y + i * line_height  # Adjust y position for each line
        text_element = dwg.text(
            line,
            insert=("50%", y_position),  # Centered horizontally
            text_anchor="middle",  # Align text center
            font_size=font_size,
            font_family="Helvetica",
            fill="black"
        )
        dwg.add(text_element)

    # Save the SVG
    dwg.save()

#--------------------------------------------------------------------------------------------------
def generate_flashcards(canv, png_files, captions, numbering_offset, total_cards, blank_png):
    """
        Draw two pages on a canvas.
        Four cards fit on a duplex sheet.
        The first page has the front side of the cards.
        The second page has the back side of the cards.
        Each side of a card has one or two diagrams.    
    """

    get_first_png = lambda lst: lst[0] if lst else blank_png
            
    # 'double_44/0004_f_2.svg' -> '0004'
    cards = sorted(list(set([ os.path.split(x)[-1].split('_')[0] for x in png_files ])))
    if len(cards) % 2 != 0:
        cards.append('xxxx')
        
    # Arrange front side (first page)
    scale_canvas(canv)
    draw_cutting_marks(canv)
    for i, card in enumerate(cards):
        front_dia_1 = get_first_png([ x for x in png_files if x.endswith(f"{card}_f_1.png") ]) # top
        front_dia_2 = get_first_png([ x for x in png_files if x.endswith(f"{card}_f_2.png") ]) # bot
        row = i // 2
        col = i % 2

        draw_png_on_canvas(canv, front_dia_1, 2 * row, col, caption = captions.get(front_dia_1, ''))
        draw_png_on_canvas(canv, front_dia_2, 2 * row + 1, col, 
                            footer = f'{i+numbering_offset}/{total_cards-1}',
                            caption = captions.get(front_dia_2, ''))
        
    canv.showPage()

    # Arrange back side (second page)
    scale_canvas(canv)
    draw_cutting_marks(canv)
    for i, card in enumerate(cards):
        try:
            backcard = cards[i+1] if i%2 == 0 else cards[i-1]
        except:
            backcard = 'xxxx'
        back_dia_1 = get_first_png([ x for x in png_files if x.endswith(f"{backcard}_b_1.png") ]) # top
        back_dia_2 = get_first_png([ x for x in png_files if x.endswith(f"{backcard}_b_2.png") ]) # bot

        row = i // 2
        col = i % 2

        draw_png_on_canvas(canv, back_dia_1, 2 * row, col, caption = captions.get(back_dia_1, ''))
        draw_png_on_canvas(canv, back_dia_2, 2 * row + 1, col, caption = captions.get(back_dia_2, ''))
                           
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

#---------------------------------------------------------------------------
def batch_convert_svgs(svg_folder, output_folder, scale):
    """Convert SVG files to PNG."""
    
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        for filename in os.listdir(svg_folder):
            if not filename.endswith(".svg"): continue
            width, height = get_svg_size(os.path.join(svg_folder, filename))
            svg_path = os.path.join(svg_folder, filename)
            output_path = os.path.join(output_folder, filename.replace(".svg", ".png"))

            # Read SVG content
            with open(svg_path, "r") as f:
                svg_content = f.read()
                
            svg_data_uri = svg_to_data_uri(svg_content)  
            
            html = f'''
            <html>
            <body style="margin:0; padding:0; background-color:white;">
            <img src="{svg_data_uri}" width={width * scale}px">
            </body>
            </html>
            '''
            page.set_content(html)

            # Get image dimensions dynamically
            bbox = page.evaluate("document.querySelector('img').getBoundingClientRect();")
            page.set_viewport_size({
                "width": int(bbox["width"]),
                "height": int(bbox["height"])
            })
            # Take a screenshot
            page.screenshot(path=output_path, omit_background=True)
            print(f"Converted: {filename} at {output_path}")
            
        # Close the browser
        browser.close()
        
#--------------------------------------------------------------------------------
def draw_png_on_canvas(canvas, png_file, row, col, footer = '', caption = ''):
    """
    Position Go diagram on the ReportLab canvas.
    row in (0,1,2,3) 
    col in (0,1)
    y goes from bottom to top
    """
    caption_height = 0
    if caption: caption_height = 10
    png_width, png_height = get_png_size(png_file)
    card_height = PAGE_HEIGHT / 2
    middle = PAGE_HEIGHT / 2
    card_width = PAGE_WIDTH / 2
    hgap = PAGE_WIDTH * 0.05
    vgap = card_height * 0.05
    botmarg = card_height * 0.05
    topmarg = card_height * 0.05
    diagram_width = (PAGE_WIDTH - 2 * hgap) / 2
    diagram_height = (card_height - botmarg - topmarg - vgap) / 2 - caption_height
    x = hgap / 2 + col * (diagram_width + hgap)
    if row == 0: y = PAGE_HEIGHT - topmarg - diagram_height 
    elif row == 1: y = PAGE_HEIGHT - topmarg - vgap - 2 * diagram_height - caption_height
    elif row == 2: y = middle - topmarg - diagram_height
    else: y = middle - topmarg - vgap - 2 * diagram_height - caption_height
                 
    #rect = (x, y, diagram_width, diagram_height)
    #draw_rect(canvas, rect)
    # Scale and center the diagram on the rectangle
    scale_x = diagram_width / png_width
    scale_y = diagram_height / png_height
    scale = min(scale_x, scale_y)
    if '0000_' in png_file: scale = max(scale_x, scale_y) # Title card
    #scale = scale_y
    
    dx = (diagram_width - png_width*scale) / 2
    dy = (diagram_height - png_height*scale) / 2
    
    canvas.drawImage(png_file, x + dx, y + dy, width=png_width*scale, height=png_height*scale)
    add_caption(canvas, caption, card_width, y+dy, col, caption_height)
    
    if footer and not footer.startswith('0/'):
        # Draw footer
        font = "Helvetica"
        size = 10
        textwidth = canvas.stringWidth(footer, font, size)
        canvas.setFont(font, size)
        canvas.drawString(card_width * col + card_width/2 - textwidth/2, 
                            (3 - row) / 2 * card_height + botmarg * 0.4, 
                            footer)

#-------------------------------------------------------------------------------
def add_caption(canvas, caption, card_width, dia_bottom, col, caption_height):
    if not caption: return
    font = "Helvetica"
    size = 10
    textwidth = canvas.stringWidth(caption, font, size)
    canvas.setFont(font, size)
    canvas.drawString(card_width * col + card_width/2 - textwidth/2, 
                        dia_bottom - caption_height, 
                        caption)

#--------------------------------    
def get_png_size(png_file):
    img = Image.open(png_file)
    width, height = img.size
    return width, height
        
#-------------------------------------------------------------        
def get_svg_size(svg_file):
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
    translate_x = (LETTER_WIDTH - scaled_width) / 2
    translate_y = (LETTER_HEIGHT - scaled_height) / 2
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
        
#-------------------------------------------------------------
def svg_to_data_uri(svg_content):
    """Convert SVG content to a base64 data URI"""
    svg_base64 = base64.b64encode(svg_content.encode("utf-8")).decode("utf-8")
    return f'data:image/svg+xml;base64,{svg_base64}'
        
#-------------------------------------------------------------        
def get_caption(txt):
    """ Parse the text between <caption> and </caption> from txt """
    start = txt.find('<caption>')
    if start == -1: return ''
    end = txt.find('</caption>')
    if end == -1: return ''
    res = txt[start+9:end].strip()
    return res
    
main()