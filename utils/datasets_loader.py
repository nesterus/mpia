import requests
import urllib
import json
import shutil
from urllib.parse import urlencode

def download_datasets(config_path, datasets_dir): 
    # config_path = './configs/datasets_links.json'
    # datasets_dir = './datasets/'
    '''
    download datasets through the links and unzip them
    in the target directory
    '''

    with open(config_path, 'r') as f:
        config = json.loads(f.read())

    for dataset_name in config["datasets_links"].keys():
        public_key = datasets_links[dataset_name] # shared link with dataset
        base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'

        final_url = base_url + urlencode(dict(public_key=public_key))
        response = requests.get(final_url)
        download_url = response.json()['href']

        destination = datasets_dir + dataset_name + '.zip' # path and name for archive with dataset
        urllib.request.urlretrieve(download_url, destination)

    for dataset_name in datasets_links.keys():
        archive_format = 'zip'
        filename = datasets_dir + dataset_name + '.' + archive_format
        # Unpack the archive file
        shutil.unpack_archive(filename, datasets_dir, archive_format)