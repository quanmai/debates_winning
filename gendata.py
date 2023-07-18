from utils.reader import generate_data, generate_data_3_splits
from utils.config import config
# from utils.data_reader import get_stats

if __name__ == "__main__":

    with open('title.txt','r') as rf:
        titles = rf.readlines()
    
    # titles = titles[:200]
    titles = [t.replace('\n','') for t in titles]
    # titles = ['Animal-Testing-Joke-Debate/1/']
    # titles = ['Jesus-probably-did-not-exist/2/']
    # titles = ['Official-TOC-Round-1/1/']
    # titles = ['Resolved-The-Death-Penalty-is-a-just-punishment-for-convicted-murderers-killers./1/']
    # titles = ['Same-sex-unions-should-not-be-federally-recognized-as-marriage./1/']
    # titles = ['Age-Of-Empires-is-better-than-Civilization/1/']
    # titles = ['Astrology-should-be-recognized-as-a-science./1/']
    # titles = ['Asking-for-a-new-9-11-investigation-is-highly-unreasonable/1/']
    # titles = ['Gays-should-be-allowed-to-marry./1/']
    generate_data(titles)
    # get_stats()
