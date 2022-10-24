import pandas as pd
from time import sleep
import requests
from bs4 import BeautifulSoup
import numpy as np
import os
import re

SOURCE_DIR = os.getcwd() + "/"
yap_token = "04e27131bcc82593e1ab56f10a493f90"


class Sentence:
    def __init__(self, text):
        self.text = text
        self.empty = False
        self.clean_text = None
        self.tokens = None
        self.morphemes = None
        self.lemmas = None

    def cleanup(self):
        text = self.text.replace('\s', ' ')
        pattern = re.compile(r'[\-\\\־]')
        text = re.sub(pattern, ' ', text).strip()
        pattern = re.compile(r'[^א-ת\s0-9]')
        self.clean_text = re.sub(pattern, '', text).strip()
        self.empty = (len(self.clean_text) == 0) or (self.clean_text.isnumeric()) or (' ' not in self.clean_text)

    def yap_split(self, return_tokens=False, return_morphemes=False, return_lemmas=False):
        if self.morphemes is None and not self.empty:
            url = f'https://www.langndata.com/api/heb_parser?token={yap_token}'
            _json = '{"data":"' + self.clean_text + '"}'
            headers = {'content-type': 'application/json'}
            sleep(3)
            r = requests.post(url, data=_json.encode('utf-8'),
                              headers={'Content-type': 'application/json; charset=utf-8'})
            json_obj = r.json()
            # print(json_obj.keys())
            try:
                if return_tokens:
                    self.tokens = json_obj["tokenized_text"]  # todo delete what I don't use
                if return_morphemes:
                    self.morphemes = json_obj["segmented_text"]
                if return_lemmas:
                    self.lemmas = json_obj["lemmas"]
            except:
                print(self.clean_text)
                try:
                    print(json_obj["message"])
                except:
                    print(json_obj["msg"])
        # return self.tokens, self.morphemes, self.lemmas
        return self.morphemes


class Post:
    def __init__(self, author=None, party=None, date=None, text=""):
        self.author = author
        self.party = party
        self.date = date
        self.text = text
        self.sentences = []
        self.token_sentences = None
        self.morph_sentences = None
        self.lemma_sentences = None

    def split_to_sentences(self, split_sentences=False):
        sentences = [Sentence(s) for s in re.split('[!.,?\n\t:()]', self.text)]
        for s in sentences:
            s.cleanup()
            if not s.empty:
                self.sentences.append(s)
        if split_sentences:
            self.split_sentences()

    def split_sentences(self):
        # split = np.array([s.yap_split(return_morphemes=True) for s in self.sentences]).transpose()
        # self.token_sentences = split[0]
        # self.morph_sentences = split[1]
        # self.lemma_sentences = split[2]
        self.morph_sentences = [s.yap_split(return_morphemes=True) for s in self.sentences]


def create_df(post):
    """
    create data frame for one url, with columns author, party, date, text.
    """
    try:
        details = post.contents[1].find_all("a", class_="link-on-blue")
        author = details[0].text.strip()
        party = details[1].text.strip()
        date = details[2].text.strip()
        post_text = post.contents[3].contents[1].text
    except:
        print("error in create_df")
        print()
        return None
    cur_dict = {"author": [author],
                "party": [party],
                "date": [date],
                "text": [post_text]}

    cur_df = pd.DataFrame(cur_dict)
    return cur_df


def create_data_file(search_word=None, file_name="0", start=1, end=0):
    """
    create csv file from kikar website (https://kikar.org), with columns author, party, date, text.
    """
    lst = []
    next_page = [0]
    i = start
    while len(next_page) != 0 and i != end:
        sleep(1)
        if search_word is not None:
            url = "https://kikar.org/search/?page=" + str(i) + "&search_str=" + search_word
        else:
            url = "https://kikar.org/?page=" + str(i)
        res = requests.get(url, verify=False)
        xml = res.text
        soup = BeautifulSoup(xml, features="html.parser")
        posts = soup.find_all("div", class_="panel panel-primary status-panel")
        for post in posts:
            df = create_df(post)
            if df is None:
                continue
            lst.append(df)
        next_page = soup.find_all("div", class_="endless_container")
        print("done:", i)
        i += 1

    total_df = pd.concat(lst)
    out_path = SOURCE_DIR + "data/" + file_name + ".csv"
    total_df.to_csv(out_path, index=False, encoding="utf-8")


def create_morphemes_file(data_file, out_file, start=0, size=0):
    data = pd.read_csv(SOURCE_DIR + "data/" + data_file + ".csv", encoding="utf-8")
    if size > 0:
        data = data[start:start + size]
    elif start > 0:
        data = data[start:]
    temp = {'author': data['author'].values,
            'party': data['party'].values,
            'date': data['date'].values}
    new_df = pd.DataFrame(temp, columns=['author', 'party', 'date'])
    posts = data['text'].values

    for i in range(len(posts)):
        temp = Post(text=posts[i])
        temp.split_to_sentences(split_sentences=True)
        posts[i] = temp.morph_sentences
        print(posts[i])
    new_df['text'] = posts
    new_df.to_csv(SOURCE_DIR + "data/" + out_file + str(start) + "_" + str(start + size) + ".csv", index=False,
                  encoding="utf-8")
    new_df.to_json(SOURCE_DIR + "data/" + out_file + str(start) + "_" + str(start + size) + ".json")
    print("*********\nDone with file " + out_file + "\n*********")


def convert_txt_to_csv(data_file, out_file, txt_file=''):
    data = pd.read_csv(SOURCE_DIR + "data/" + data_file + ".csv", encoding="utf-8")

    temp = {'author': data['author'].values,
            'party': data['party'].values,
            'date': data['date'].values}
    new_df = pd.DataFrame(temp, columns=['author', 'party', 'date'])

    f = open(SOURCE_DIR + "data/" + txt_file + ".txt", encoding="utf-8")
    posts = f.readlines()[25:]
    for i in range(len(posts)):
        post = posts[i]
        post = post.split("\', \'")
        post[0] = post[0][2:]
        post[-1] = post[-1][:-3]
        posts[i] = post
        print(posts[i])
    new_df['text'] = posts
    new_df.to_csv(SOURCE_DIR + "data/" + out_file + ".csv", index=False,
                  encoding="utf-8")
    new_df.to_json(SOURCE_DIR + "data/" + out_file + ".json")


def convert_and_join_csv_to_json(data_files, out_file, shuffle=False):
    dfs = []
    for data_file in data_files:
        data = pd.read_csv(SOURCE_DIR + "data/" + data_file + ".csv", encoding="utf-8")

        temp = {'author': data['author'].values,
                'party': data['party'].values,
                'date': data['date'].values}
        new_df = pd.DataFrame(temp, columns=['author', 'party', 'date'])

        posts = data['text'].values
        for i in range(len(posts)):
            post = posts[i]
            post = post.split("\', \'")
            post[0] = post[0][2:]
            post[-1] = post[-1][:-3]
            posts[i] = post
        new_df['text'] = posts
        dfs.append(new_df)
    out_df = pd.concat(dfs, ignore_index=True)
    if shuffle:
        out_df.sample(frac=1).reset_index(drop=True)  # shuffle rows
    out_df.to_csv(SOURCE_DIR + "data/" + out_file + ".csv", index=False, encoding="utf-8")
    out_df.to_json(SOURCE_DIR + "data/" + out_file + ".json")


def clean(data_file, out_file):
    data = pd.read_json(SOURCE_DIR + "data/" + data_file + ".json", encoding="utf-8")

    temp = {'author': data['author'].values,
            'party': data['party'].values,
            'date': data['date'].values}
    new_df = pd.DataFrame(temp, columns=['author', 'party', 'date'])

    posts = data['text'].values

    for i in range(len(posts)):
        post = posts[i]
        posts[i] = np.array([s for s in post if ' ' in s])
        print(posts[i])
    new_df['text'] = posts
    new_df.to_csv(SOURCE_DIR + "data/" + out_file + ".csv", index=False, encoding="utf-8")
    new_df.to_json(SOURCE_DIR + "data/" + out_file + ".json")


def shuffle(data_file):
    data = pd.read_json(SOURCE_DIR + "data/" + data_file + ".json", encoding="utf-8")
    data = data.sample(frac=1).reset_index(drop=True)  # shuffle rows
    data.to_csv(SOURCE_DIR + "data/" + data_file + "_shuffled.csv", index=False, encoding="utf-8")
    data.to_json(SOURCE_DIR + "data/" + data_file + "_shuffled.json")


def join_jsons(data_files, out_file, shuffle=False):
    dfs = []
    for data_file in data_files:
        data = pd.read_json(SOURCE_DIR + "data/" + data_file + ".json", encoding="utf-8")
        dfs.append(data)

    out_df = pd.concat(dfs, ignore_index=True)
    if shuffle:
        out_df.sample(frac=1).reset_index(drop=True)  # shuffle rows
    out_df.to_csv(SOURCE_DIR + "data/" + out_file + ".csv", index=False, encoding="utf-8")
    out_df.to_json(SOURCE_DIR + "data/" + out_file + ".json")




if __name__ == '__main__':
    create_morphemes_file("trump/trump", "trump/trump_morph", 500, 100)

    a = 2
