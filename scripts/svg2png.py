
import base64
from playwright.sync_api import sync_playwright
import xml.etree.ElementTree as ET
import os,re
import argparse
from pdb import set_trace as BP

#---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Convert SVG files to PNG")
    parser.add_argument("svg_folder", help="Folder containing SVG files")
    parser.add_argument("output_folder", help="Output folder for PNG files")
    parser.add_argument("--scale", type=int, default=1.0, help="Scale factor for the output PNG")
    args = parser.parse_args()

    batch_convert_svgs(args.svg_folder, args.output_folder, args.scale)

#---------------------------------------------------------------------------
def batch_convert_svgs(svg_folder, output_folder, scale):
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        for filename in os.listdir(svg_folder):
            if not filename.endswith(".svg"): continue
            width, height = get_svg_dimensions(os.path.join(svg_folder, filename))
            svg_path = os.path.join(svg_folder, filename)
            output_path = os.path.join(output_folder, filename.replace(".svg", ".png"))

            # Read SVG content
            with open(svg_path, "r") as f:
                svg_content = f.read()
                
            svg_data_uri = svg_to_data_uri(svg_content)  
            
            html = f'''
            <html>
            <body style="margin:0; padding:0;">
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

#-------------------------------------------------------------        
def get_svg_dimensions(svg_file):
    """Extracts width and height from the <svg> tag in an SVG file."""
    tree = ET.parse(svg_file)
    root = tree.getroot()
    width = root.attrib.get("width")
    height = root.attrib.get("height")

    return float(width), float(height)

#-------------------------------------------------------------
def svg_to_data_uri(svg_content):
    """Convert SVG content to a base64 data URI"""
    svg_base64 = base64.b64encode(svg_content.encode("utf-8")).decode("utf-8")
    return f'data:image/svg+xml;base64,{svg_base64}'


main()
