import os
import shutil


def copytree(src, dst, symlinks=False, ignore=None):
    # https://stackoverflow.com/questions/1868714/how-do-i-copy-an-entire-directory-of-files-into-an-existing-directory-using-pyth
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)
    return


def save_parameter_dict(filename, dictionary):
    with open(filename, 'w') as f:
        for key, value in dictionary.items():
            f.write('%s : %s\n' % (key, value))
    return
