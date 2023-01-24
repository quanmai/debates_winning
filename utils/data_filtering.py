import json
import os
from functools import reduce
from config import config

def filter_no_text(debates):
    '''
        filter out debates that one side says nothing
    '''
    debate_dict = {}
    count = 0
    for key, debate in debates.items():
        text = ['','']
        for r in debate['rounds']:
            for side in r:
                if side['side'] == 'Pro':
                    text[0] += side['text']
                else:
                    text[1] += side['text']
        if len(text[0]) == 0 or len(text[1]) == 0:
            count += 1
            continue
        else:
            debate_dict[key] = debate
    
    print('there are {} conversations that have one side muted'.format(count))
    return debate_dict

def filter_short_arguments(debates, k=5):
    '''
        Keep only the ones with at least 5 sentences 
        by each debater in each round
    '''

    debate_dict = {}
    count = 0 

    for key, debate in debates.items():
        flag = False
        for round in debate['rounds']:
            for side in round:
                if len(side['text'].split('.')) < k:
                    flag = True
        if not flag:
            debate_dict[key] = debate
        else:
            count += 1

    print('There are {} debates that have less than {} sentence in each round'.format(count,k))
    
    return debate_dict

                    

def filter_tied(debates):
    '''
        filter out tied debates
    '''

    debate_dict = {}
    count = 0
    for key, debate in debates.items():
        if debate['participant_1_status'] == debate['participant_2_status']:
            count += 1
            continue
        else:
            debate_dict[key] = debate

    print('there are {} conversations that tied'.format(count))

    return debate_dict

def filter_tied_argument(debates):
    '''
        # filter out debates that are tied on Made more convincing arguments
    '''
    debate_generate = {}

    for key,debate in debates.items():
        temp_pro = 0
        temp_con = 0
        Pro = debate['participant_1_name']
        Con = debate['participant_2_name']
        for user in debate['votes']:
            for name, attitude in user['votes_map'].items():
                if 'Made more convincing arguments' in attitude:
                    if name == Pro and attitude['Made more convincing arguments'] == True:
                        temp_pro = temp_pro + 1
                    elif name == Con and attitude['Made more convincing arguments'] == True:
                        temp_con = temp_con + 1
        if temp_pro != temp_con:
            debate_generate[key] = debate

    return debate_generate


def filter_forfeit_argument(debates):
    debate_dict = {}
    i = 0
    for key, debate in debates.items():
        flag = False
        for round in debate['rounds']:
            for side in round:
                if side['text'] == 'forfeit':
                    flag = True
        if not flag:
            debate_dict[key] = debate
        else:
            i += 1

    print('There are {} debates that one of participants forfeited'.format(i))
    
    return debate_dict

def filter_less_than_k_turns(debates, k=3):
    debate_dict = {}
    i = 0
    for key, debate in debates.items():
        if len(debate['rounds']) < k:
            i+= 1
            continue   
        else:
            debate_dict[key] = debate
    
    print('There are {} debates that have less than {} turns'.format(i,k))

    return debate_dict

def filter_lass_than_k_voters(debates, k=5):
    debate_dict = {}
    i = 0
    for key, debate in debates.items():
        if len(debate['votes']) < k:
            i+= 1
            continue   
        else:
            debate_dict[key] = debate
    
    print('There are {} debates that have less than {} voters'.format(i,k))

    return debate_dict

def filter_tied_argument_less_than_x(debates, k=1):
    debate_dict = {}
    i=0
    for key,debate in debates.items():
        temp_pro = 0
        temp_con = 0
        Pro = debate['participant_1_name']
        Con = debate['participant_2_name']
        for user in debate['votes']:
            for name, attitude in user['votes_map'].items():
                if 'Made more convincing arguments' in attitude:
                    if name == Pro and attitude['Made more convincing arguments'] == True:
                        temp_pro = temp_pro + 1
                    elif name == Con and attitude['Made more convincing arguments'] == True:
                        temp_con = temp_con + 1
        if temp_pro - temp_con > k or temp_con - temp_pro > k:
            debate_dict[key] = debate
        else:
            i += 1 

    print('There are {} debates that have less than {} different in votes'.format(i,k))
    return debate_dict

if __name__ == '__main__':
    debates = {}

    with open(config.original_f, 'r') as f:
        debates = json.load(f)
    f.close()

    # Preprocess argument
    debates = reduce(
        lambda value, function: function(value),
        (
            filter_no_text,
            filter_short_arguments,
            filter_tied,
            filter_tied_argument,
            filter_forfeit_argument,
            filter_less_than_k_turns,
            filter_lass_than_k_voters,
            filter_tied_argument_less_than_x,
        ),
        debates,
    )

    with open(config.filtered_f, 'w') as f:
        json.dump(debates, f)
    f.close()
    print(len(debates))