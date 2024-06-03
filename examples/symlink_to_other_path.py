import os, shutil
from itertools import takewhile


def main():
    files = open("/home/sheffler/project/dborigami/BIG_FILES").readlines()
    for f in files[:10]:
        print(f.split())
        try:
            i = f.find(" ")
            s, f = int(f[:i]), f[i:].strip()
        except:
            pass

        d, b = os.path.split(f)
        newd = f"/data/{d[6:]}"
        newf = f"{newd}/{b}"

        if not os.path.exists(f) or os.path.islink(f):
            print("skipping", s, f)
            print(newf)
            assert os.path.exists(newf), f"no {newf}"
        else:
            assert f.startswith("/home")
            print("moving", s, f)
            os.makedirs(newd, exist_ok=True)
            shutil.copyfile(f, f"{newd}/{b}")
            os.remove(f)
            os.symlink(newf, f)


if __name__ == "__main__":
    main()
