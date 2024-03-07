import torch

if torch.cuda.is_available():
    # make empty txt file with name yes.txt
    with open('yes.txt', 'w') as f:
        f.write('yes')
else:
    # make empty txt file with name no.txt
    with open('no.txt', 'w') as f:
        f.write('no')
