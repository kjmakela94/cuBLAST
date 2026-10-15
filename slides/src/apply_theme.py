"""Write the deck's theme colours and fonts into ppt/theme/theme*.xml (standalone; no external helpers).
Usage: python apply_theme.py deck.pptx"""
import re, sys, zipfile, shutil

THEME = dict(name="GPU Kernels", head="Cambria", body="Calibri",
             colors=dict(dk1="16202E", lt1="FFFFFF", dk2="2B3A4F", lt2="F1F4F8", accent1="76B900", accent2="3B82F6",
                         accent3="E6A817", accent4="D64545", accent5="9AA5B4", accent6="5B6577", hlink="3B82F6", folHlink="2B3A4F"))

def scheme():
    c = THEME["colors"]
    inner = "".join(f'<a:{k}><a:srgbClr val="{c[k]}"/></a:{k}>' for k in
                    ["dk1", "lt1", "dk2", "lt2", "accent1", "accent2", "accent3", "accent4", "accent5", "accent6", "hlink", "folHlink"])
    return f'<a:clrScheme name="{THEME["name"]}">{inner}</a:clrScheme>'

def patch(xml):
    xml = re.sub(r"<a:clrScheme\b.*?</a:clrScheme>", scheme(), xml, flags=re.S)
    xml = re.sub(r'(<a:majorFont>\s*<a:latin typeface=")[^"]*(")', rf'\g<1>{THEME["head"]}\g<2>', xml)
    xml = re.sub(r'(<a:minorFont>\s*<a:latin typeface=")[^"]*(")', rf'\g<1>{THEME["body"]}\g<2>', xml)
    return xml

src = sys.argv[1]; tmp = src + ".tmp"
with zipfile.ZipFile(src) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
    for it in zin.infolist():
        data = zin.read(it.filename)
        if re.match(r"ppt/theme/theme\d+\.xml$", it.filename):
            data = patch(data.decode("utf8")).encode("utf8")
        zout.writestr(it, data)
shutil.move(tmp, src)
print("theme applied to", src)
