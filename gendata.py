from utils.data_reader import generate_data_test

if __name__ == "__main__":

    with open('title.txt','r') as rf:
        titles = rf.readlines()
    
    titles = titles[:100]
    titles = [t.replace('\n','') for t in titles]
    
    generate_data_test(titles)