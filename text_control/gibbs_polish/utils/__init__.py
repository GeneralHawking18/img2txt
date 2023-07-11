import numpy as np
import os
import colorlog
import random
import torch
# from text_control.gibbs_polish.frameworks import sentiments_classifier, POS_classifier

def remove_stop_words(args, lm_tokenizer):
    with open(args.stop_words_path,'r',encoding='utf-8') as stop_words_file:
        stop_words = stop_words_file.readlines() # Đọc tất cả các stop_words từ file
        stop_words_ = [stop_word.rstrip('\n') for stop_word in stop_words] # Loại bỏ các dấu xuống dòng của các stop_word, loại bỏ các "\n"
        stop_words_ += args.add_extra_stopwords
        stop_ids = lm_tokenizer.convert_tokens_to_ids(stop_words_) # Lấy id của các stop words
        print(stop_ids)

        token_mask = torch.ones((1,lm_tokenizer.vocab_size)) # Khởi tạo một tensor 2 chiều 1x vocab_size, tât cả = 1, vì sao lại 2 chiều, bởi vì khi stack vào batch dễ hơn, 1 + 1 + 1, chứ vd mỗi V thì stack thành 10 x V khó .
        for stop_id in stop_ids:
            token_mask[0,stop_id]=0 # gán tất cá các stop_words đều gán = 0 hết
        token_mask = token_mask.to(args.device) # Cho vào gpu và cuda.
    return token_mask

def create_logger(folder, filename):
    log_colors = {
        'DEBUG': 'blue',
        'INFO': 'white',
        'WARNING': 'green',
        'ERROR': 'red',
        'CRITICAL': 'yellow',
    }

    import logging
    logger = logging.getLogger('ConZIC')
    # %(filename)s$RESET:%(lineno)d
    # LOGFORMAT = "%(log_color)s%(asctime)s [%(log_color)s%(filename)s:%(lineno)d] | %(log_color)s%(message)s%(reset)s |"
    LOGFORMAT = ""
    LOG_LEVEL = logging.DEBUG
    logging.root.setLevel(LOG_LEVEL)
    stream = logging.StreamHandler()
    stream.setLevel(LOG_LEVEL)
    stream.setFormatter(colorlog.ColoredFormatter(LOGFORMAT, datefmt='%d %H:%M', log_colors=log_colors))

    # print to log file
    hdlr = logging.FileHandler(os.path.join(folder, filename))
    hdlr.setLevel(LOG_LEVEL)
    # hdlr.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    hdlr.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(hdlr)
    logger.addHandler(stream)
    return logger

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_init_text(tokenizer, seed_text, max_len, batch_size=1):
    """ Get initial sentence by padding seed_text with [mask] words to max_len """
    text = seed_text + tokenizer.mask_token * max_len
    ids = tokenizer.encode(text)
    batch = [ids] * batch_size
    return batch

def update_token_mask(tokenizer, token_mask, max_len, index):
    """ '.'(full stop) is only allowed in the last token position """
    if index == max_len - 1:
        token_mask[:, tokenizer.vocab['.']] = 1
    else:
        token_mask[:, tokenizer.vocab['.']] = 0
    return token_mask

def format_output(sample_num, FinalCaption, BestCaption):
    if sample_num == 1:
        return f"{FinalCaption[0]}", f"{BestCaption[0]}"
    elif sample_num ==2:
        return f"{FinalCaption[0]}\n{FinalCaption[1]}", f"{BestCaption[0]}\n{BestCaption[1]}"
    elif sample_num ==3:
        return f"{FinalCaption[0]}\n{FinalCaption[1]}\n{FinalCaption[2]}",\
            f"{BestCaption[0]}\n{BestCaption[1]}\n{BestCaption[2]}"
    elif sample_num ==4:
        return f"{FinalCaption[0]}\n{FinalCaption[1]}\n{FinalCaption[2]}\n{FinalCaption[3]}",\
            f"{BestCaption[0]}\n{BestCaption[1]}\n{BestCaption[2]}\n{BestCaption[3]}"
    else:
        return f"{FinalCaption[0]}\n{FinalCaption[1]}\n{FinalCaption[2]}\n{FinalCaption[3]}\n{FinalCaption[4]}",\
            f"{BestCaption[0]}\n{BestCaption[1]}\n{BestCaption[2]}\n{BestCaption[3]}\n{BestCaption[4]}"