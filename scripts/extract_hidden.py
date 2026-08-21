import re
modules = re.findall(r'"(src\.[^"]+)"', open("src/app.py").read())
print(" ".join("--hidden-import " + m for m in modules))
