import torch
import pickle
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

##### https://github.com/githubharald/CTCDecoder/blob/master/src/BeamSearch.py
class BeamEntry:
    "information about one single beam at specific time-step"
    def __init__(self):
        self.prTotal = 0 # blank and non-blank
        self.prNonBlank = 0 # non-blank
        self.prBlank = 0 # blank
        self.prText = 1 # LM score
        self.lmApplied = False # flag if LM was already applied to this beam
        self.labeling = () # beam-labeling

class BeamState:
    "information about the beams at specific time-step"
    def __init__(self):
        self.entries = {}

    def norm(self):
        "length-normalise LM score"
        for (k, _) in self.entries.items():
            labelingLen = len(self.entries[k].labeling)
            self.entries[k].prText = self.entries[k].prText ** (1.0 / (labelingLen if labelingLen else 1.0))

    def sort(self):
        "return beam-labelings, sorted by probability"
        beams = [v for (_, v) in self.entries.items()]
        sortedBeams = sorted(beams, reverse=True, key=lambda x: x.prTotal*x.prText)
        return [x.labeling for x in sortedBeams]

    def wordsearch(self, classes, ignore_idx, beamWidth, dict_list):
        beams = [v for (_, v) in self.entries.items()]
        sortedBeams = sorted(beams, reverse=True, key=lambda x: x.prTotal*x.prText)[:beamWidth]

        for j, candidate in enumerate(sortedBeams):
            idx_list = candidate.labeling
            text = ''
            for i,l in enumerate(idx_list):
                if l not in ignore_idx and (not (i > 0 and idx_list[i - 1] == idx_list[i])):  # removing repeated characters and blank.
                    text += classes[l]

            if j == 0: best_text = text
            if text in dict_list:
                print('found text: ', text)
                best_text = text
                break
            else:
                print('not in dict: ', text)
        return best_text

def applyLM(parentBeam, childBeam, classes, lm):
    "calculate LM score of child beam by taking score from parent beam and bigram probability of last two chars"
    if lm and not childBeam.lmApplied:
        c1 = classes[parentBeam.labeling[-1] if parentBeam.labeling else classes.index(' ')] # first char
        c2 = classes[childBeam.labeling[-1]] # second char
        lmFactor = 0.01 # influence of language model
        bigramProb = lm.getCharBigram(c1, c2) ** lmFactor # probability of seeing first and second char next to each other
        childBeam.prText = parentBeam.prText * bigramProb # probability of char sequence
        childBeam.lmApplied = True # only apply LM once per beam entry

def addBeam(beamState, labeling):
    "add beam if it does not yet exist"
    if labeling not in beamState.entries:
        beamState.entries[labeling] = BeamEntry()

def ctcBeamSearch(mat, classes, ignore_idx, lm, beamWidth=25, dict_list = []):
    "beam search as described by the paper of Hwang et al. and the paper of Graves et al."

    #blankIdx = len(classes)
    blankIdx = 0
    maxT, maxC = mat.shape

    # initialise beam state
    last = BeamState()
    labeling = ()
    last.entries[labeling] = BeamEntry()
    last.entries[labeling].prBlank = 1
    last.entries[labeling].prTotal = 1

    # go over all time-steps
    for t in range(maxT):
        curr = BeamState()

        # get beam-labelings of best beams
        bestLabelings = last.sort()[0:beamWidth]

        # go over best beams
        for labeling in bestLabelings:

            # probability of paths ending with a non-blank
            prNonBlank = 0
            # in case of non-empty beam
            if labeling:
                # probability of paths with repeated last char at the end
                prNonBlank = last.entries[labeling].prNonBlank * mat[t, labeling[-1]]

            # probability of paths ending with a blank
            prBlank = (last.entries[labeling].prTotal) * mat[t, blankIdx]

            # add beam at current time-step if needed
            addBeam(curr, labeling)

            # fill in data
            curr.entries[labeling].labeling = labeling
            curr.entries[labeling].prNonBlank += prNonBlank
            curr.entries[labeling].prBlank += prBlank
            curr.entries[labeling].prTotal += prBlank + prNonBlank
            curr.entries[labeling].prText = last.entries[labeling].prText # beam-labeling not changed, therefore also LM score unchanged from
            curr.entries[labeling].lmApplied = True # LM already applied at previous time-step for this beam-labeling

            # extend current beam-labeling
            for c in range(maxC - 1):
                # add new char to current beam-labeling
                newLabeling = labeling + (c,)

                # if new labeling contains duplicate char at the end, only consider paths ending with a blank
                if labeling and labeling[-1] == c:
                    prNonBlank = mat[t, c] * last.entries[labeling].prBlank
                else:
                    prNonBlank = mat[t, c] * last.entries[labeling].prTotal

                # add beam at current time-step if needed
                addBeam(curr, newLabeling)

                # fill in data
                curr.entries[newLabeling].labeling = newLabeling
                curr.entries[newLabeling].prNonBlank += prNonBlank
                curr.entries[newLabeling].prTotal += prNonBlank

                # apply LM
                #applyLM(curr.entries[labeling], curr.entries[newLabeling], classes, lm)

        # set new beam state
        last = curr

    # normalise LM scores according to beam-labeling-length
    last.norm()

    # sort by probability
    #bestLabeling = last.sort()[0] # get most probable labeling

    # map labels to chars
    #res = ''
    #for idx,l in enumerate(bestLabeling):
    #    if l not in ignore_idx and (not (idx > 0 and bestLabeling[idx - 1] == bestLabeling[idx])):  # removing repeated characters and blank.
    #        res += classes[l]

    if dict_list == []:
        bestLabeling = last.sort()[0] # get most probable labeling
        res = ''
        for i,l in enumerate(bestLabeling):
            if l not in ignore_idx and (not (i > 0 and bestLabeling[i - 1] == bestLabeling[i])):  # removing repeated characters and blank.
                res += classes[l]
    else:
        res = last.wordsearch(classes, ignore_idx, beamWidth, dict_list)

    return res
#####

def consecutive(data, mode ='first', stepsize=1):
    group = np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
    group = [item for item in group if len(item)>0]

    if mode == 'first': result = [l[0] for l in group]
    elif mode == 'last': result = [l[-1] for l in group]
    return result

def word_segmentation(mat, separator_idx =  {'th': [1,2],'en': [3,4]}, separator_idx_list = [1,2,3,4]):
    result = []
    sep_list = []
    start_idx = 0
    for sep_idx in separator_idx_list:
        if sep_idx % 2 == 0: mode ='first'
        else: mode ='last'
        a = consecutive( np.argwhere(mat == sep_idx).flatten(), mode)
        new_sep = [ [item, sep_idx] for item in a]
        sep_list += new_sep
    sep_list = sorted(sep_list, key=lambda x: x[0])

    for sep in sep_list:
        for lang in separator_idx.keys():
            if sep[1] == separator_idx[lang][0]: # start lang
                sep_lang = lang
                sep_start_idx = sep[0]
            elif sep[1] == separator_idx[lang][1]: # end lang
                if sep_lang == lang: # check if last entry if the same start lang
                    new_sep_pair = [lang, [sep_start_idx+1, sep[0]-1]]
                    if sep_start_idx > start_idx:
                        result.append( ['', [start_idx, sep_start_idx-1] ] )
                    start_idx = sep[0]+1
                    result.append(new_sep_pair)
                else: # reset
                    sep_lang = ''

    if start_idx <= len(mat)-1:
        result.append( ['', [start_idx, len(mat)-1] ] )
    return result

class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    #def __init__(self, character, separator = []):
    def __init__(self, character, separator_list = {}, dict_pathlist = {}):
        # character (str): set of the possible characters.
        dict_character = list(character)

        #special_character = ['\xa2', '\xa3', '\xa4','\xa5']
        #self.separator_char = special_character[:len(separator)]

        self.dict = {}
        #for i, char in enumerate(self.separator_char + dict_character):
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'blank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[blank]'] + dict_character  # dummy '[blank]' token for CTCLoss (index 0)
        #self.character = ['[blank]']+ self.separator_char + dict_character  # dummy '[blank]' token for CTCLoss (index 0)
        self.separator_list = separator_list

        separator_char = []
        for lang, sep in separator_list.items():
            separator_char += sep

        self.ignore_idx = [0] + [i+1 for i,item in enumerate(separator_char)]

        dict_list = {}
        for lang, dict_path in dict_pathlist.items():
            with open(dict_path, "rb") as input_file:
                word_count = pickle.load(input_file)
            dict_list[lang] = word_count
        self.dict_list = dict_list

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]

        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode_greedy(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]

            char_list = []
            for i in range(l):
                if t[i] not in self.ignore_idx and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank (and separator).
                #if (t[i] != 0) and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank (and separator).
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += l
        return texts

    def decode_beamsearch(self, mat, beamWidth=5):
        texts = []

        for i in range(mat.shape[0]):
            t = ctcBeamSearch(mat[i], self.character, self.ignore_idx, None, beamWidth=beamWidth)
            texts.append(t)
        return texts

    def decode_wordbeamsearch(self, mat, beamWidth=5):
        texts = []
        argmax = np.argmax(mat, axis = 2)
        for i in range(mat.shape[0]):
            words = word_segmentation(argmax[i])
            string = ''
            for word in words:
                matrix = mat[i, word[1][0]:word[1][1]+1,:]
                if word[0] == '': dict_list = []
                else: dict_list = self.dict_list[word[0]]
                t = ctcBeamSearch(matrix, self.character, self.ignore_idx, None, beamWidth=beamWidth, dict_list=dict_list)
                string += t
            texts.append(string)
        return texts

class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res













import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import datetime as dt
import time
import pandas.io.sql as psql
import psycopg2 as pg
import timeit
import operator
import re
from db import *
from tqdm import tqdm


def portal_login(username, password):
    logout = requests.get("https://portal.aimazing.co/dashboard/logout")
    payload = {"username": username, "password": password}    
    res = requests.post("https://portal.aimazing.co/dashboard/login", params=payload)
    return res

def get_receipt_image(did, cookies):
    payload = {
        "id": did,
        "entry_type" : 'raster'
    }
    res = requests.get(
        "https://portal.aimazing.co/dashboard/api/receipt_parsing/receiptData/image",
        params=payload,
        cookies=login.cookies
    )
    with open(f"{did}.png", 'wb') as f:
            f.write(res.content)
    return 

def get_text_original(img_64, lang = 'eng2', preprocessors = ['filter-receipt'], psm = '6', preprocessor_args = {}):
    payload = {
        "img_base64": img_64,
        "engine": "tesseract",
        "engine_args": {
            "config_vars": { "tessedit_do_invert": "0", "page_separator": "" },
            "lang": lang,
            "psm": psm
        },
        "preprocessors": preprocessors,
        "preprocessor-args": preprocessor_args
    }
    res = requests.post(
        "http://13.114.216.220:9292/ocr",
        json=payload
    )
    return res

def get_text_enhanced(img_64, lang = 'eng2', preprocessors = 'filter-receipt', psm = ''):
    payload = {
        "img_base64": img_64,
        "engine": "tesseract",
        "engine_args": {
            "config_vars": { "tessedit_do_invert": "0", "page_separator": "" },
            "lang": lang,
            "psm": psm,
            "dpi": "75"
        },
        "preprocessors": ["filter-receipt"],
        "preprocessor-args": {}
    }
    #payload = json.dumps(payload)
    res = requests.post(
        "http://13.114.216.220:9292/ocr",
        json=payload
    )
    return res

def print_format(img_64, lang = 'eng2', preprocessors = 'filter-receipt', psm = ''):
    payload = {
        "img_base64": 'img_64',
        "engine": "tesseract",
        "engine_args": {
            "config_vars": { "tessedit_do_invert": "0", "page_separator": "" },
            "lang": lang,
            "psm": psm,
            "dpi": "75"
        },
        "preprocessors": ["filter-receipt"],
        "preprocessor-args": {}
    }
    return payload

login = portal_login(config["USER"]["NAME"], config["USER"]["PSWD"])

def parsing_rule_by_outlet(id):
    qstr = f"""
    select 
        a.outlet_name,
        c.pattern as rule_group_name,
        d.field as receipt_type,
        a.outlet_id,
        a.rule_group_id,
        c.field,
        c.regexp,
        c.header_cut_point, 
        c.footer_cut_point
    from (
        select
            name as outlet_name,
            owner_outlet_id as outlet_id,
            rule_group_id
        from device_info
    ) a 
    inner join 
    group_use_parsing_rule b
    on a.rule_group_id = b.group_id
    inner join 
        (
        select 
            id,
            field, 
            pattern, 
            regexp, 
            header_cut_point, 
            footer_cut_point,
            receipt_type_rule_id
        from parsing_rule
        where exist is true
    ) c
    on b.rule_id = c.id
    inner join (select field, id from parsing_rule) d
    on c.receipt_type_rule_id = d.id
    where outlet_id = {id}
    order by receipt_type
    """
    return aimbox_rds.query(text=qstr)

def get_receipts(id, type_of_receipt):
    if type_of_receipt == 'closed bill':
        qstr = f"""
        WITH transactions AS (
        SELECT 
            max(id) AS id
        FROM 
            aimbox
        WHERE 
            outlet_id = {id}
            AND created_at >= now() - INTERVAL '3 months' 
            AND type = '{type_of_receipt}'
        GROUP BY 
            outlet_id, workdate, receipt_id 
        ),
        trans_base AS (
        SELECT 
            * 
        FROM (
            SELECT      
                id,      
                outlet_id,      
                receipt_id,      
                date,      
                time,      
                text AS total,     
                created_at,     
                device_timestamp,     
                detail ->> 'subtotal' AS subtotal,     
                detail ->> 'GST' AS gst,     
                detail ->> 'Service Charge' AS service_charge,     
                detail ->> 'receipt_discount' AS receipt_discount,     
                detail ->> 'rounding' AS rounding,     
                detail ->> 'tips' AS tips,
                raw_text
                FROM 
                aimbox a    
            WHERE EXISTS (     
                SELECT * FROM transactions WHERE a.id = id   
            )     
        ) a   
        JOIN LATERAL (
            SELECT 
                MAX(id) AS tally_id   
            FROM 
                aimbox_data_health adh    
            WHERE      
                aimbox_id = a.id   
        ) b   
        ON TRUE 
        )  
        SELECT    
            t.id, 
            t.outlet_id, 
            t.receipt_id, 
            adh.has_total,   
            adh.has_receipt_id,   
            adh.has_workdate,   
            adh.valid_monetary,   
            adh.valid_formula_total,   
            adh.valid_payment_total,   
            adh.valid_rounding,  
            t.date, 
            t.time, 
            t.total, 
            t.subtotal, 
            t.gst, 
            t.service_charge, 
            t.receipt_discount, 
            t.rounding,
            t.raw_text
        FROM 
            trans_base t 
        JOIN 
            aimbox_data_health adh 
        ON 
            t.tally_id = adh.id 
        """
        return aimbox.query(text=qstr)
    elif type_of_receipt == 'sales day report':
        qstr = f"""
        SELECT
            receipt_id,
            outlet_id, 
            work_date,
            content::json -> 'reportTotalRevenue' as sdr_total,
            content::json -> 'dashboardTotalRevenue' as dashboard_total,
            content::json -> 'dashboardTransactions' as num_transaction
        FROM 
            outlet_data_event 
        WHERE 
            outlet_id = {id}
        ORDER BY
            work_date

        """
        return aimbox_rds.query(text=qstr)

def get_string_from_match(line):
    try:
        return line[0][0]
    except IndexError as error:
        print('no line found')
        return 'NULL'

def get_value_from_match(line):
    try:
        return line[0][1]
    except IndexError as error:
        print('no regex round')
        return ''
        #raise IndexError('No regex match found.')

def get_regex_text(regex, res_enhanced):
    matched_line = re.findall(rf'({regex})', res_enhanced)
    final_string = re.sub(rf'{get_value_from_match(matched_line)}', '',get_string_from_match(matched_line))
    return final_string

def get_captured_value(regex, res_enhanced):
    matched_line = re.findall(rf'({regex})', res_enhanced)
    final_string = get_value_from_match(matched_line)
    return final_string

def get_accuracy_field(df, field, captured_group): #df, time_text, time_captured_group
    final = df.groupby([field]).agg({'did':"count", captured_group:'nunique'}).reset_index()
    final = final.assign(accuracy_col = lambda x: x[captured_group].apply(lambda y: y/sum(x[captured_group]) if y == max(x[captured_group]) else 0)).accuracy_col.sum()
    return final





#JABEZ's changes "1 day"-> "3 day"
def get_all_sdr(date, outlets):
    qstr = f"""
    with suntec_outlets as (
        select id, name
        from outlet
        where id in {outlets}
        ),

    dates as (
        SELECT generate_series
        ( '{date}'::date + interval '8 hours'
            , '{date}'::date  + interval '8 hours'
            , '3 day'::interval)::date as date
    ),

    suntec_outlet_info as (
        select *
        from suntec_outlets a
        cross join dates b
        where id in {outlets}
        ),

    ranked_sdr as (
        select 
        id as outlet_id, 
        name,
        date,
        receipt_id as did, 
        status, 
        timestamp, 
        content::json, 
        difference, 
        trans_diff,
        row_number() over (partition by outlet_id, date,receipt_id order by date, timestamp desc) as ranks
        from (
            select distinct *
            from suntec_outlet_info a 
            left join (select receipt_id, work_date, outlet_id, 
                        status, timestamp, content, difference, trans_diff 
                        from outlet_data_event) b 
            on a.date = b.work_date and a.id = b.outlet_id
        ) sub
    )

    select 
        outlet_id, 
        name,
        date, 
        did,
        content ->> 'reportTotalRevenue' as sdr_total_revenue
    from ranked_sdr
    where (ranks = 1) or (ranks >= 2 and did is null)
    order by outlet_id, date
    """
    return aimbox_rds.query(text=qstr)

#def get_all_sdr_from_aimbox(date = 'now()'):
#matter of approach: when unclear as to what is the correct SDR, is the latest did the correct one, or the 
#did with the closest amount to the closed bill amount?
###no value add for now because workdate is used as query instead of created_at 
def get_closed_bills_total(outlet_id, date):
    qstr = f"""
    with trans_base as(
        SELECT 
            max(id) AS id
        FROM 
            aimbox
        WHERE 
            outlet_id = {outlet_id}
        AND 
            workdate = '{date}'
        AND type = 'closed bill'   
        GROUP BY 
            outlet_id, workdate, receipt_id 
    )

    select sum(CASE
                WHEN REGEXP_REPLACE(text, '[0-9,]+\.\d{{1,2}}', '', 'g') = '' then text::float
                else 0
                end) as closeddbill_total_revenue, 
            sum(CASE
                WHEN REGEXP_REPLACE(text, '[0-9,]+\.\d{{1,2}}', '', 'g') = '' then 1
                else 0
                end) as closeddbill_transaction_counts,
            workdate as date, 
            outlet_id
    from aimbox a
    join trans_base b 
    on a.id = b.id
    group by workdate, outlet_id
    """
    return aimbox.query(text=qstr)

def get_sdr_created_at(outlet_id, did):
    qstr = f"""
    select id as did, (created_at + interval '8 hours')::date as created_at
    from aimbox
    where id = '{did}'
    and outlet_id = {outlet_id}
    """
    return aimbox.query(text=qstr)

def get_closed_bill_mode(outlet_id, created_at_date, did):
    qstr = f"""
    SELECT 
    mode() WITHIN GROUP (ORDER BY date) as closedbill_mode_workdate,
    outlet_id,
    {did} as did,
    '{created_at_date}'::date as date
    from aimbox
    where outlet_id = {outlet_id}
    and datetime BETWEEN '{created_at_date}'::date 
    AND '{created_at_date}'::date + interval '33 hours'

    and created_at + interval '8 hours' BETWEEN '{created_at_date}'::date 
    AND '{created_at_date}'::date + interval '33 hours'
    and type = 'closed bill'
    group by outlet_id, did, '{created_at_date}'::date
    """
    return aimbox.query(text=qstr)


def get_sdr(sdr_did):
    qstr = f"""
        SELECT
            outlet_id, 
            work_date,
            content::json -> 'reportTotalRevenue' as sdr_total,
            content::json -> 'dashboardTotalRevenue' as dashboard_total,
            content::json -> 'dashboardTransactions' as num_transaction
        FROM 
            outlet_data_event 
        WHERE 
            receipt_id = {sdr_did}
        ORDER BY
            work_date

        """
    return aimbox_rds.query(text=qstr)

def get_closed_bill_workdate(outlet_id):
    qstr = f"""
        SELECT
            date
        FROM 
            aimbox 
        WHERE 
            outlet_id = {outlet_id}
        AND
            type = 'closed bill'
        AND
            created_at BETWEEN now() AND now() - interval '1 day'

        """
    return aimbox.query(text=qstr)

def sdr_workdate_check(sdr_workdate, maj_closed_bill_workdate,current_did):
    if str(sdr_workdate) == str(maj_closed_bill_workdate):
        return int(current_did)
    else:
        return "Error! Check " + str(current_did)

def extract_dids_for_outlet(outlet_id,did_col_name, gross_sales_sdrs_col_name, gross_sales_closed_bills_col_name, workdate,df):
    df_temp = df[(df.outlet_id == outlet_id) & (df.workdate == workdate)]
    sdr_did_list = df_temp[did_col_name].tolist()
    gross_sales_sdrs = df_temp[gross_sales_sdrs_col_name].tolist()
    gross_sales_closed_bills = df_temp[gross_sales_closed_bills_col_name].tolist()
    gross_sales_closed_bills = gross_sales_closed_bills[0]
    #print(sdr_did_list, outlet_id)
    return sdr_did_list, gross_sales_sdrs,gross_sales_closed_bills

def function1(current_did):
    # SDR
    data_sdr = get_sdr(current_did).drop_duplicates().reset_index()
    sdr_workdate = str(data_sdr.work_date[0])
    outlet_id = (data_sdr.outlet_id)[0]


    # Closed Bill
    closed_bill_workdate = get_closed_bill_workdate(outlet_id)
    maj_closed_bill_workdate = closed_bill_workdate.mode()
    maj_closed_bill_workdate = str(maj_closed_bill_workdate['date'][0])
    return sdr_workdate, maj_closed_bill_workdate

def function2(gross_sales_sdrs):
    if gross_sales_sdrs.count(gross_sales_sdrs[0]) == len(gross_sales_sdrs):
        return "total collection of all SDRs is the same"
    else:
        return "total collection of SDRs are different"

def function3(current_did,index_pos_current_did,gross_sales_sdrs):
    if gross_sales_sdrs[index_pos_current_did] == max(gross_sales_sdrs):
        return "latest sdr has highest total collection"
        
    return "proceed to check 4"



#changed function 4 to smallest margin
def function4(sdr_did_list,gross_sales_closed_bills,gross_sales_sdrs,margin=0.10):    
    within_margin=[]
    for i in gross_sales_sdrs:
        float_i = float(i)
        if margin*gross_sales_closed_bills>=abs(gross_sales_closed_bills-float_i):
            index_pos2=gross_sales_sdrs.index(i)
            within_margin.append([abs(gross_sales_closed_bills-float_i),sdr_did_list[index_pos2]])
    within_margin.sort()

    if len(within_margin)==0:
        return sdr_did_list[0]
    else:
        # print(within_margin[0][1])
        # print("4")
        return within_margin[0][1]



# def function4(sdr_did_list,gross_sales_closed_bills,gross_sales_sdrs,margin=0.10):    
#     within_margin=[]
#     for i in gross_sales_sdrs:
#         float_i = float(i)
#         if margin*gross_sales_closed_bills>=abs(gross_sales_closed_bills-float_i):
#             index_pos2=gross_sales_sdrs.index(i)
#             within_margin.append([float(i),sdr_did_list[index_pos2]])
#     within_margin.sort(reverse=True)
#     if len(within_margin)==0:
#         return "FALSE, no suitable SDRs found"
#     else:
#         return within_margin[0][1]



def sdr_error_detect(outlet_id, workdate,df):   
    
    ##defining variables(optional functionality)
    did_col_name = "did"
    gross_sales_closed_bills_col_name = "closeddbill_total_revenue"
    gross_sales_sdrs_col_name = "sdr_total_revenue"
    
    # query to output a list of sdrs within the threshold for the specific outlet id
    sdr_did_list, gross_sales_sdrs, gross_sales_closed_bills = extract_dids_for_outlet(
        outlet_id,did_col_name,
        gross_sales_sdrs_col_name,
        gross_sales_closed_bills_col_name,
        workdate,df)
    
    if len(sdr_did_list)==1 and sdr_did_list[0]==0: #assuming that len(sdr_did_list) will never be 0 due to the above code
        return "FALSE, no suitable SDRs found"
    else:
        for i in range(0,len(sdr_did_list)):
            current_did = sdr_did_list[i]
            ## Function 1
            sdr_workdate = str(df.query(did_col_name+'=='+str(current_did))['workdate'].iloc[0])
            maj_closed_bill_workdate = str(df.query(did_col_name+'=='+str(current_did))['closedbill_mode_workdate'].iloc[0])
            results_from_func_1 = sdr_workdate_check(sdr_workdate, maj_closed_bill_workdate,current_did)
            output=[0,0,0,0]
            output[1]=function2(gross_sales_sdrs)
            output[2]=function3(sdr_did_list[0],i,gross_sales_sdrs)
            output[3]=function4(sdr_did_list,gross_sales_closed_bills,gross_sales_sdrs)
            # if sdr is wrong, move on to next sdr 
            if isinstance(results_from_func_1,int) == False:
                # print(results_from_func_1)
                # terminates checking if this is the last SDR in the loop. UPDATE: no point since we r querying using workdate now. 
                if i == len(sdr_did_list) - 1:
                    return "wrong_workdate error, no suitable SDRs found"
                continue
            # return SDR as correct if this is the last SDR with correct workdate in the loop, terminates check
            # limitation: if only one sdr has correct workdate, but the value is also wrong, they reprint next week with the right value
            # will not be able to detect and verify
            elif i == len(sdr_did_list)-1:
                return current_did
            else:
                # print(output[1])
                if output[1] == "total collection of all SDRs is the same":
                    print(str(current_did)+"total collection values of all SDRs captured are same")
                    return current_did
                elif output[1] == "total collection of SDRs are different":
                    # print(output[2])
                    if output[2] == "latest sdr has highest total collection":
                        print(str(current_did)+"latest higherst")    
                        return current_did
                    else:
                        print("output from function 4, "+str(output[3]))
                        return output[3]



def get_current_system_sdr(date, outlets):
        qstr = f"""
        with suntec_outlets as (
            select id, name
            from outlet
            where id in {outlets}
            ),

        dates as (
            SELECT generate_series
            ( '{date}'::date
             , '{date}'::date 
             , '1 day'::interval)::date as date
        ),

        suntec_outlet_info as (
            select *
            from suntec_outlets a
            cross join dates b
            where id in {outlets}
            ),

        ranked_sdr as (
            select 
            id as outlet_id, 
            name,
            date, 
            receipt_id as did, 
            status, 
            timestamp, 
            content::json, 
            difference, 
            trans_diff,
            row_number() over (partition by outlet_id, date order by date, timestamp desc) as ranks
            from (
                select distinct *
                from suntec_outlet_info a 
                left join (select receipt_id, work_date, outlet_id, 
                           status, timestamp, content, difference, trans_diff 
                           from outlet_data_event) b 
                on a.date = b.work_date and a.id = b.outlet_id
            ) sub
        )

        select 
            outlet_id, 
            name,
            date as workdate, 
            did as system_did
       --     content ->> 'reportTotalRevenue' as sdr_total_revenue
        from ranked_sdr
        where (ranks = 1) --or (ranks >= 2 and did is null)
        and did is not null
       order by outlet_id, date
        """
        return aimbox_rds.query(text = qstr)


def get_updated_suntec_list(outlets): 
    qstr = f"""
    select id as outlet_id, name as outlet_name
    from outlet
    where id in {outlets}
    """
    return aimbox_rds.query(text = qstr)
    




#for i in range(diff_in_days.days+1):

def query_from_db(date_to_query, outlets):
    

    ## Get SDR amount 
    print('Raymond first step')
    all_did_of_sdr = get_all_sdr(date_to_query, outlets)
    print(all_did_of_sdr)
    all_did_of_sdr['did'] = list(all_did_of_sdr.did.fillna(0).astype(int))
    all_did_of_sdr['sdr_total_revenue'] = list(all_did_of_sdr.sdr_total_revenue.fillna(0).astype(float))

    ## add closed bill total
    print('Raymond second step')
    #drop_duplicated_workdate = all_did_of_sdr[['outlet_id', 'date']].drop_duplicates().reset_index(drop = True)
    final_closed_bill_total = pd.DataFrame()
    for i in range(len(all_did_of_sdr)):
        final_closed_bill_total = pd.concat([get_closed_bills_total(all_did_of_sdr.outlet_id[i], all_did_of_sdr.date[i]), final_closed_bill_total])
    final_combined_step2 = all_did_of_sdr.merge(final_closed_bill_total[['outlet_id', 'date', 'closeddbill_total_revenue']], on = ['outlet_id', 'date'], how = 'left').drop_duplicates().reset_index(drop = True)
    final_combined_step2['closeddbill_total_revenue'] = final_combined_step2['closeddbill_total_revenue'].fillna(0)
    
    #add SDR created_at
    print('Raymond third step')
    all_sdr_created_at = pd.DataFrame()
    for i in range(len(final_combined_step2)):
        all_sdr_created_at = pd.concat([get_sdr_created_at(final_combined_step2.outlet_id[i], final_combined_step2.did[i]), all_sdr_created_at])
    final_combined_step3 = final_combined_step2.merge(all_sdr_created_at, on = 'did', how = 'left')

    final_kept = final_combined_step3[final_combined_step3.created_at.isna()].reset_index(drop = True)

    final_combined_step4 = final_combined_step3.dropna()
    final_combined_step4 = final_combined_step4.reset_index(drop = True)
    
    # add closed billmode workdate
    print('Raymond fourth step')
    closed_bill_workdate = pd.DataFrame()
    for i in range(len(final_combined_step4)):
        #JABEZ's changes final_combined_step4.created_at[i]->final_combined_step4.date[i]
        closed_bill_workdate = pd.concat([get_closed_bill_mode(final_combined_step4.outlet_id[i], final_combined_step4.date[i], final_combined_step4.did[i]), closed_bill_workdate])
    final_combined = final_combined_step4.merge(closed_bill_workdate, on = ['outlet_id', 'date', 'did'], how = 'left')
    final_kept['closedbill_mode_workdate'] = final_kept['date']
    final_combined = pd.concat([final_combined, final_kept])
    #JABEZ's changes
    final_combined['closedbill_mode_workdate'] = final_combined['closedbill_mode_workdate'].fillna(date_to_query)
    final_combined= final_combined.rename(columns={'date':'workdate'})
    
    
    return final_combined




def SDR_automation(df):
    #df[['outlet_id','sdr_workdate']].drop_duplicates().groupby(['outlet_id'])['sdr_workdate'].apply(lambda x: ', '.join(x) ).reset_index()                    
    final = df[['outlet_id', 'name', 'workdate']].drop_duplicates().groupby(['outlet_id', 'name'])['workdate'].apply(list).reset_index()
    #workdates = final["workdate"].drop_duplicates().to_list()
    #workdates = df['sdr_workdate'].drop_duplicates().tolist()
    
    
    ##creating output for automated_mechanism
    final_list = []
    for i in range(len(final)):
        for k in range(len(final.workdate[i])):
           # print(final.outlet_id[i], final.workdate[i])
            to_be_added = (final.outlet_id[i], final.name[i], final.workdate[i][k], sdr_error_detect(final.outlet_id[i], final.workdate[i][k],df))
            final_list.append(to_be_added)
    code_output = pd.DataFrame(final_list, columns = ['outlet_id', 'name', 'workdate', 'did'])
    
    return code_output


#adding outlets with no SDRs    
def combining_workdates_with_no_sdrs(code_output,date_to_query,list_of_outlets,final_df_output,df):
    df_current_output = code_output
    date_check = date_to_query
    #df2 = pd.read_csv('All Available Tenants in Suntec - outlet_202108201125.csv')
    df2 = get_updated_suntec_list(list_of_outlets)
    list_of_outlets=list_of_outlets

    outlet_id_1 = list(df_current_output['outlet_id'].drop_duplicates())
    new_list = list(set(list_of_outlets).difference(outlet_id_1))
    final_output2 = pd.DataFrame(columns = ['outlet_id', 'workdate', 'did'])
    final_output2["outlet_id"] = new_list
    final_output_sheet = pd.merge(final_output2,df2, on= ["outlet_id"], how="inner")

    final_output_sheet = final_output_sheet.rename(columns={'outlet_name':'name'}) 
    final_all_outlets = pd.concat([code_output,final_output_sheet])
    final_all_outlets['workdate'] = final_all_outlets['workdate'].fillna(date_to_query)
    ###for testing only
    # final_all_outlets.to_csv('test.csv',index=False)
    
    # may not be very useful
    final_all_outlets = final_all_outlets.fillna('Missing DID')
    
    
    system_sdr = get_current_system_sdr(date_to_query, list_of_outlets)
    final_output = final_all_outlets.merge(system_sdr, on = ['outlet_id', 'workdate', 'name'], how = 'left')
    final_output = final_output.merge(df[['did', 'sdr_total_revenue']].rename(columns={'sdr_total_revenue':'system_sdr_amount', 'did':'system_did'}), on = 'system_did', how = 'left')
    final_output = final_output.merge(df[['did', 'sdr_total_revenue']].rename(columns={'sdr_total_revenue':'code_sdr_amount'}), on = 'did', how = 'left')
    final_output = final_output.merge(df[['outlet_id', 'workdate', 'closeddbill_total_revenue']], on = ['outlet_id', 'workdate'], how = 'left').rename(columns={'did':'code_did'})
    final_output = final_output.drop_duplicates().reset_index(drop = True)
    final_output = final_output.fillna('Missing')
    final_output.to_csv(f'SDR_Auto_Error_Detection_{date_to_query}.csv',index=False)

    #final concatenation
    final_output.to_csv(f'SDR_Error_Detection_Cashback_Suntec_Outlets_{date_check}.csv',index=False)
    
    final_df_output = pd.concat([final_df_output, final_output])  
    
    return final_df_output



def convert_to_int(x):
    result = np.nan
    try:
        result = int(float(x))
    except TypeError:
        pass
    finally:
        return result

def convert_to_float(x):
    result = np.nan
    try:
        result = float(x)
    except TypeError:
        pass
    finally:
        return result
        
def same_did(code,system):
    if code == 'FALSE, no suitable SDRs found' or system == 'Missing':
        return 'Missing' 
    else: 
        if float(code) == float(system):
            return True
        else :
            return False

def same_amt(code,system):
    if code == 'FALSE, no suitable SDRs found' or system == 'Missing':
        return 'Missing' 
    else: 
        if float(code) == float(system):
            return True 
        else : 
            return False 

def captured_did_correctly(did, amt):
    if did == True:
        if bool(did) == amt:
            return True
        else:
            return "" 
    else: 
        return ""

def remove_suntec(x):
    if x[-9:] == ' @ Suntec':
        return x[:-9].rstrip()
    elif x[-14:] == ' @ Suntec City':
        return x[:-14].rstrip()
    else:
        return x.rstrip()
    
    
#validation_formula
def validation(amt):
    if isinstance(amt,str) == False:
        return float(amt)/1.07
    else:
        return amt



#small diff
def threshold(api, validation):
    if abs(api-validation)<0.3:
        return True
    else: 
        return False




def valid_03(total_report_amt, api_value):
    if float(api_value) == 0:
        return 'FALSE'
    elif abs(float(total_report_amt)-float(api_value))<=0.3:
        return 'TRUE'
    else:
        return 'FALSE'


    
def valid_gst(sdr_total_revenue, api_value):
    if float(api_value) == 0.0:
        return 'FALSE'
    elif abs(float(sdr_total_revenue)/1.07-float(api_value))<=0.3:
        return 'TRUE'
    else:
        return 'FALSE'

    
def check(df, calculated_rev, total_revenue):
    if df['type'][i] == 'special_gst' or df['type'][i] == 'gst':
        df['Good To Go'][i] = valid_gst(calculated_rev, total_revenue)
        df['Negligible Difference'][i] = valid_gst(calculated_rev, total_revenue)
        df['Validation'][i] = calculated_rev/1.07
    else:
        df['Good To Go'][i] = valid_03(calculated_rev, total_revenue)
        df['Negligible Difference'][i] = valid_03(calculated_rev, total_revenue)
        df['Validation'][i] = calculated_rev    

        
        
        
        
def get_total_report_amt(sdr_did, s, login_url, login_data):
    url = 'https://portal.aimazing.co/dashboard/api/info_management/outlets/outletDataEventSummary?id='+sdr_did
    s.post(login_url, data = login_data)
    res = s.get(url).json()
    total = 0 
    if 'tally_checking' in res:
        if 'payment_breakdown' in res['tally_checking']:
            paym=res['tally_checking']['payment_breakdown']
            for i in range(len(paym)):
                if 'report_amount' in paym[i]:
                    total = total + paym[i]['report_amount']
        #         print(paym[i]['payment'])
        #         print(paym[i]['report_amount'])
    #     print(sdr_did)
    #     print(total)
    #     print()
    return total



def get_tax(sdr_did, s, login_url, login_data):
    url = 'https://portal.aimazing.co/dashboard/api/info_management/outlets/outletDataEventSummary?id='+sdr_did
    s.post(login_url, data = login_data)
    res = s.get(url).json()
    total = 0 
    tax=0
    if 'other_tax' in res['tally_checking']:
        tax=res['tally_checking']['other_tax']
    return tax
        
        

        