
from pdb import set_trace as BP
import argparse
import http.server
import json
import os
import shutil
import socketserver
import tempfile
import webbrowser

#-----------------
def main():
    parser = argparse.ArgumentParser(description='Browse through svg flashcards in a folder')
    parser.add_argument('--folder', type=str, help='folder containing the SVG files', required=True)
    args = parser.parse_args()

    blank_svg = f'{args.folder}/blank.svg'
    generate_blank_svg(blank_svg)

    svg_files = [f'{args.folder}/{f}' for f in os.listdir(args.folder) if f.endswith('.svg')] 
    cards = sorted(list(set([ os.path.split(x)[-1].split('_')[0] for x in svg_files ])))
    cards = [x for x in cards if not 'blank' in x]

    svg_pairs = []
    for idx, card in enumerate(cards):
        front_1 = f"{args.folder}/{card}_f_1.svg" if os.path.exists(f"{args.folder}/{card}_f_1.svg") else blank_svg
        front_2 = f"{args.folder}/{card}_f_2.svg" if os.path.exists(f"{args.folder}/{card}_f_2.svg") else blank_svg
        back_1 = f"{args.folder}/{card}_b_1.svg" if os.path.exists(f"{args.folder}/{card}_b_1.svg") else blank_svg
        back_2 = f"{args.folder}/{card}_b_2.svg" if os.path.exists(f"{args.folder}/{card}_b_2.svg") else blank_svg
        if front_1 != blank_svg or front_2 != blank_svg: svg_pairs.append((front_1, front_2))
        if back_1 != blank_svg or back_2 != blank_svg: svg_pairs.append((back_1, back_2)) 

    htmldir = os.path.join(args.folder, "html")
    os.makedirs(htmldir, exist_ok=True)
    indexhtml = generate_html_files(svg_pairs, htmldir)
    serve_html(htmldir, indexhtml)

#-------------------------------------------------------------        
def generate_blank_svg(fname): 
    with open(fname, 'w') as f:
        f.write('<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="100" height="100"></svg>')

#----------------------------------------------        
def generate_html_files(svg_pairs, folder):
    svg_pairs_basename = [(os.path.basename(pair[0]), os.path.basename(pair[1])) for pair in svg_pairs]
    svg_pairs_json = json.dumps(svg_pairs_basename)
    viewer = os.path.join(folder, "index.html")
    with open(viewer, "w") as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>SVG Viewer</title>
  <style>
    body { text-align: center; font-family: sans-serif; }
    img { width: 45%%; margin: 1%%; border: 1px solid #ccc; }
  </style>
</head>
<body>
  Press the left and right arrow keys to navigate through the cards.
  <div id="image-container">
  <img src="0020_f_1.svg"><img src="0020_f_2.svg">
  </div>
  <script>
    const svgPairs = %s;
    let index = 0;

    function render() {
      const pair = svgPairs[index];
      document.getElementById("image-container").innerHTML = 
        `<img src="${pair[0]}"><img src="${pair[1]}">`;
      console.log("Rendering pair:", pair);
      document.title = `Pair ${index + 1} of ${svgPairs.length}`;
    }

    document.addEventListener("keydown", (e) => {
      if (e.key === "ArrowRight" && index < svgPairs.length - 1) {
        index++; render();
      } else if (e.key === "ArrowLeft" && index > 0) {
        index--; render();
      }
    });

    render(); 
  </script>
</body>
</html>
""" % str(svg_pairs_json))
    # Copy the SVG files into the temp folder
    for pair in svg_pairs:
        for svg in pair:
            shutil.copy(svg, os.path.join(folder, os.path.basename(svg)))

    return viewer

#----------------------------------------------
def serve_html(folder, html_file):
    os.chdir(folder)
    port = 8000
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        webbrowser.open(f"http://localhost:{port}/{os.path.basename(html_file)}")
        print(f"Serving at http://localhost:{port}/")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("Server stopped.")

if __name__ == "__main__":
    main()
