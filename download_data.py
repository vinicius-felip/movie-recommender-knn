from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

def main():
    folder=Path(__file__).resolve().parent/'data'
    folder.mkdir(exist_ok=True)
    archive=folder/'ml-100k.zip'
    urlretrieve('https://files.grouplens.org/datasets/movielens/ml-100k.zip',archive)
    with ZipFile(archive) as z:
        for name in ('u.data','u.item','README'):
            z.extract('ml-100k/'+name,folder)
    archive.unlink()
    print('Base disponível em',folder/'ml-100k')
if __name__=='__main__': main()
