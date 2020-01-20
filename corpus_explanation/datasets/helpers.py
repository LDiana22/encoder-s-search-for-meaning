import torch
SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True



def get_indices_tensor(text_arr, word_to_indx, max_length):
    '''
    -text_arr: array of word tokens
    -word_to_indx: mapping of word -> index
    -max length of return tokens

    returns tensor of same size as text with each words corresponding
    index
    '''
    nil_indx = 0
    text_indx = [ word_to_indx[x] if x in word_to_indx else nil_indx for x in text_arr][:max_length]
    if len(text_indx) < max_length:
        text_indx.extend( [nil_indx for _ in range(max_length - len(text_indx))])

    x =  torch.LongTensor([text_indx])

    return x