import os, pathlib, shutil
from multiprocessing import Pool

def change_class(file_path, new_location = None, old_class = None, new_class = None):
    # filename = pathlib.Path(file_path).name
    with open(file_path, 'r') as f1:
        texts = f1.readlines()
        new_text = ""
        for text in texts:
            print(text)
            if new_class is not None:
                if int(text.split(' ')[0]) == int(old_class):
                    # print(text.split(' ')[1:])
                    new_text += new_class + ' ' + ' '.join(text.split(' ')[1:])
                else:
                    new_text += text
            else:
                new_text += text
        f1.close()

    if new_location:
        with open(new_location, 'w') as f2:
            f2.write(new_text)
            f2.close()
    else:
        with open(file_path, 'w') as f2:
            # print(new_text)
            f2.write(new_text)
            f2.close()

