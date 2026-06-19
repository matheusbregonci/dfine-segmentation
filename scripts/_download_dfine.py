import urllib.request, zipfile, io, os, shutil

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dest = os.path.join(root, "D-FINE-seg")

data = None
for branch in ("main", "master"):
    url = f"https://github.com/ArgoHA/D-FINE-seg/archive/refs/heads/{branch}.zip"
    try:
        print("Baixando", url)
        data = urllib.request.urlopen(url, timeout=180).read()
        print("OK", len(data), "bytes")
        break
    except Exception as e:
        print("falhou", branch, e)
        data = None

if not data:
    raise SystemExit("download falhou")

z = zipfile.ZipFile(io.BytesIO(data))
tmp = os.path.join(root, "_dfine_tmp")
if os.path.exists(tmp):
    shutil.rmtree(tmp)
z.extractall(tmp)
inner = os.path.join(tmp, os.listdir(tmp)[0])
if os.path.exists(dest):
    shutil.rmtree(dest)
shutil.move(inner, dest)
shutil.rmtree(tmp)
print("Extraido em", dest)
