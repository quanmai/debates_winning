from utils.data_reader import generate_data
from utils.config import config

if __name__ == "__main__":

    with open('title.txt','r') as rf:
        titles = rf.readlines()
    
    titles = titles[100:200]
    titles = [t.replace('\n','') for t in titles]
    # titles = ['3-Items-found-at-Wal-Mart-that-will-best-kill-a-velociraptor/1/']
    # titles = ['A-debate-on-a-certain-controversy/1/']
    generate_data(titles, file_name='dataset/dataset_preproc_100_200.p')