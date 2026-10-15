"""Post-process the roofline scatter chart that pptxgenjs cannot fully style:
log-scale X axis, drop blank points, per-series line/marker styles.  Usage: python postprocess_roofline.py deck.pptx"""
import re, sys, zipfile, shutil, os

src = sys.argv[1]; tmp = src + ".tmp"
zin = zipfile.ZipFile(src)
target = None
for n in zin.namelist():
    if n.startswith("ppt/charts/chart") and n.endswith(".xml") and b"Roof (measured)" in zin.read(n):
        target = n
assert target, "roofline chart not found"
x = zin.read(target).decode("utf8")

# 1) remove blank points (pptxgenjs writes <c:v></c:v> for nulls)
x = re.sub(r'<c:pt idx="\d+"><c:v></c:v></c:pt>', "", x)
# 2) log-scale the X (first) value axis
first = x.index("<c:valAx>")
s_i = x.index("<c:scaling>", first) + len("<c:scaling>")
if "<c:logBase" not in x[s_i:x.index("</c:scaling>", s_i)]:
    x = x[:s_i] + '<c:logBase val="10"/>' + x[s_i:]
# 3) per-series styling
SYMB = {1: None, 2: "circle", 3: "diamond", 4: "square", 5: "triangle"}
def style(m):
    ser = m.group(0)
    idx = int(re.search(r'<c:idx val="(\d+)"/>', ser).group(1)) + 1   # 1 = roof
    color = re.search(r'<a:srgbClr val="([0-9A-Fa-f]{6})"', ser).group(1)
    sp_start = ser.index("<c:spPr>"); sp_end = ser.index("</c:spPr>") + len("</c:spPr>")
    mk_start = ser.index("<c:marker>"); mk_end = ser.index("</c:marker>") + len("</c:marker>")
    assert sp_end <= mk_start
    if SYMB[idx] is None:   # roof: thick line, no markers
        sp = f'<c:spPr><a:ln w="38100" cap="rnd"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:round/></a:ln></c:spPr>'
        mk = '<c:marker><c:symbol val="none"/></c:marker>'
    else:                   # points: no connecting line, filled marker with white rim
        sp = '<c:spPr><a:ln><a:noFill/></a:ln></c:spPr>'
        size = 9 if idx == 2 else 12
        mk = (f'<c:marker><c:symbol val="{SYMB[idx]}"/><c:size val="{size}"/><c:spPr><a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
              f'<a:ln w="12700"><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill></a:ln></c:spPr></c:marker>')
    return ser[:sp_start] + sp + ser[sp_end:mk_start] + mk + ser[mk_end:]
x = re.sub(r"<c:ser>.*?</c:ser>", style, x, flags=re.S)

with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data = x.encode("utf8") if item.filename == target else zin.read(item.filename)
        zout.writestr(item, data)
zin.close(); shutil.move(tmp, src)
print("post-processed", target)
