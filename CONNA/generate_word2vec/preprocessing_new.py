from os.path import join
import os
import sys

sys.path.append("..")
import codecs
import json
import math
from collections import defaultdict as dd
from embedding import EmbeddingModel
from datetime import datetime
# from utils.cache import LMDBClient
from utils import data_utils
from utils import feature_utils
from utils import string_utils
from utils import settings
from utils import multithread_utils
import numpy as np
from nltk.corpus import stopwords

start_time = datetime.now()

_pubs_dict = None


def get_pub_feature(i):
    if i % 1000 == 0:
        print("The %dth paper" % i)
    pid = list(_pubs_dict)[i]
    paper = _pubs_dict[pid]
    if "title" not in paper or "authors" not in paper:
        return None
    if len(paper["authors"]) > 300:
        return None
    if len(paper["authors"]) > 30:
        print(i, pid, len(paper["authors"]))
    n_authors = len(paper.get('authors', []))
    authors = []
    for j in range(n_authors):
        author_features, word_features = feature_utils.extract_author_features(paper, j)
        aid = '{}-{}'.format(pid, j)
        authors.append((aid, author_features, word_features))
    return authors


def dump_pub_features_to_file():
    """
    generate author features by raw publication data and dump to files
    author features are defined by his/her paper attributes excluding the author's name
    """
    global _pubs_dict

    # Load publication features
    # _pubs_dict = data_utils.load_json('./WhoIsWho_data', 'conna_pub_dict.json')
    _pubs_dict = data_utils.load_json('./na-check-new', 'paper_dict_used_mag_for_conna.json')
    res = multithread_utils.processed_by_multi_thread(get_pub_feature, range(len(_pubs_dict)))
    data_utils.dump_data(res, "Essential_Embeddings_new/", "pub.features")
    # _pubs_dict = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'pubs_raw.json')
    # res = multithread_utils.processed_by_multi_thread(get_pub_feature, range(len(_pubs_dict)))
    # data_utils.dump_data(res, settings.GLOBAL_DATA_DIR, "pub.features")


def cal_feature_idf():
    """
    calculate word IDF (Inverse document frequency) using publication data
    """
    features = data_utils.load_data('Essential_Embeddings_new/', "pub.features")
    feature_dir = join('Essential_Embeddings_new/', 'global')
    os.makedirs(feature_dir, exist_ok=True)
    index = 0
    author_counter = dd(int)
    author_cnt = 0
    word_counter = dd(int)
    word_cnt = 0
    none_count = 0
    for pub_index in range(len(features)):
        pub_features = features[pub_index]
        # print(pub_features)
        if (pub_features == None):
            none_count += 1
            continue
        for author_index in range(len(pub_features)):
            aid, author_features, word_features = pub_features[author_index]

            if index % 100000 == 0:
                print(index, aid)
            index += 1

            for af in author_features:
                author_cnt += 1
                author_counter[af] += 1

            for wf in word_features:
                word_cnt += 1
                word_counter[wf] += 1

    author_idf = {}
    for k in author_counter:
        author_idf[k] = math.log(author_cnt / author_counter[k])

    word_idf = {}
    for k in word_counter:
        word_idf[k] = math.log(word_cnt / word_counter[k])

    data_utils.dump_data(dict(author_idf), feature_dir, "author_feature_idf.pkl")
    data_utils.dump_data(dict(word_idf), feature_dir, "word_feature_idf.pkl")
    print("None count: ", none_count)


_emb_model = None


def get_feature_index(i):
    word = _emb_model.wv.index2word[i]
    embedding = _emb_model.wv[word]
    return (i, embedding)


def dump_emb_array(emb_model, output_name):
    global _emb_model
    _emb_model = emb_model
    # transform the feature embeddings from embedding to (id, embedding)
    res = multithread_utils.processed_by_multi_thread(get_feature_index, range(len(_emb_model.wv.vocab)))
    sorted_embeddings = sorted(res, key=lambda x: x[0])
    word_embeddings = list(list(zip(*sorted_embeddings))[1])
    data_utils.dump_data(np.array(word_embeddings), 'Essential_Embeddings_new/emb/', output_name)


def get_feature_ids_idfs_for_one_pub(features, emb_model, idfs):
    id_list = []
    idf_list = []
    for feature in features:
        if not feature in emb_model.wv:
            continue
        id = emb_model.wv.vocab[feature].index
        idf = 1
        if idfs and feature in idfs:
            idf = idfs[feature]
        id_list.append(id)
        idf_list.append(idf)
    return id_list, idf_list


def dump_feature_id_to_file():
    """
    transform a publication into a set of author and word IDs, dump it to csv
    """
    model = EmbeddingModel.Instance()
    author_emb_model = model.load_author_name_emb()
    author_emb_file = "author_emb.array"
    word_emb_model = model.load_word_name_emb()
    word_emb_file = "word_emb.array"
    dump_emb_array(author_emb_model, author_emb_file)
    dump_emb_array(word_emb_model, word_emb_file)

    features = data_utils.load_data('Essential_Embeddings_new/', "pub.features")
    author_idfs = data_utils.load_data('Essential_Embeddings_new/global/', 'author_feature_idf.pkl')
    word_idfs = data_utils.load_data('Essential_Embeddings_new/global/', 'word_feature_idf.pkl')
    index = 0
    feature_dict = {}
    for pub_index in range(len(features)):
        pub_features = features[pub_index]
        if (pub_features == None):
            continue
        for author_index in range(len(pub_features)):
            aid, author_features, word_features = pub_features[author_index]
            if index % 100000 == 0:
                print(index, author_features, word_features)
            index += 1
            author_id_list, author_idf_list = get_feature_ids_idfs_for_one_pub(author_features, author_emb_model,
                                                                               author_idfs)
            word_id_list, word_idf_list = get_feature_ids_idfs_for_one_pub(word_features, word_emb_model, word_idfs)

            if author_id_list is not None or word_id_list is not None:
                feature_dict[aid] = (author_id_list, author_idf_list, word_id_list, word_idf_list)
    data_utils.dump_data(feature_dict, 'Essential_Embeddings_new/emb/', "pub_feature.ids")


def gen_paper_dict_for_conna():
    paper_dict = data_utils.load_json("/home/zfj/research-data/na-checking/aminer-new1", "paper_dict_used_mag.json")
    paper_dict_new = {}

    mag_venue_id_to_name = {}
    with open("/home/zfj/research-data/oag-2-1/mag_venues.txt") as rf:
        for i, line in enumerate(rf):
            cur_v = json.loads(line)
            mag_venue_id_to_name[cur_v["id"]] = cur_v["NormalizedName"]

    for i, pid in enumerate(paper_dict):
        if i % 10000 == 0:
            print("paper", i)
        cur_paper_dict_old = paper_dict[pid]
        cur_paper_dict = {"id": pid}
        if "title" in cur_paper_dict_old:
            cur_paper_dict["title"] = cur_paper_dict_old["title"]
        authors_new = []
        authors = cur_paper_dict_old.get("authors", [])
        authors_sorted = sorted(authors, key=lambda x: x["AuthorSequenceNumber"])
        for a in authors_sorted:
            a_dict_new = {"name": a.get("OriginalAuthor")}
            if a.get("OriginalAffiliation", "").strip():
                a_dict_new["org"] = a.get("OriginalAffiliation", "").strip()
            authors_new.append(a_dict_new)
        cur_paper_dict["authors"] = authors_new
        vid = cur_paper_dict_old.get("venue_id")
        if vid and vid in mag_venue_id_to_name:
            v_name = mag_venue_id_to_name[vid]
            cur_paper_dict["venue"] = v_name
        keywords = [x.get("name") for x in cur_paper_dict_old.get("fos", [])]
        if len(keywords) > 0:
            cur_paper_dict["keywords"] = keywords
        cur_paper_dict["year"] = cur_paper_dict_old.get("year")
        paper_dict_new[pid] = cur_paper_dict

    out_dir = "./na-check-new/"
    os.makedirs(out_dir, exist_ok=True)
    data_utils.dump_json(paper_dict_new, out_dir, "paper_dict_used_mag_for_conna.json")


def gen_train_name_aid_to_pids():
    name_aid_to_pids = data_utils.load_json("/home/zfj/research-data/na-checking/aminer-new1/", "aminer_name_aid_to_mag_pids_merge.json")
    name_aid_to_pids_new = dd(dict)
    paper_dict = data_utils.load_json("./na-check-new", "paper_dict_used_mag_for_conna.json")

    process_paper_cnt = 0
    for i, name in enumerate(name_aid_to_pids):
        print("name", i, name)
        cur_name_dict = name_aid_to_pids[name]
        for j, aid in enumerate(cur_name_dict):
            cur_pids = cur_name_dict[aid]
            pids_new = []
            for pid in cur_pids:
                if process_paper_cnt % 10000 == 0:
                    print("process paper cnt", process_paper_cnt)
                pid = str(pid)
                if pid not in paper_dict:
                    continue
                for a_i, a in enumerate(paper_dict[pid].get("authors", [])):
                    cur_a_name = a["name"]
                    cur_a_name = string_utils.clean_name(cur_a_name)
                    if cur_a_name == name:
                        pid_order = str(pid) + "-" + str(a_i)
                        pids_new.append(pid_order)
                        break
                process_paper_cnt += 1
            if len(pids_new) >= 5:
                name_aid_to_pids_new[name][aid] = pids_new

    data_utils.dump_json(name_aid_to_pids_new, "./na-check-new", "train_author_pub_index_profile.json")
    data_utils.dump_json(name_aid_to_pids_new, "./na-check-new", "train_author_pub_index_test.json")


def gen_test_name_aid_to_pids():
    name_aid_to_pids_new = data_utils.load_json("./na-check-new", "train_author_pub_index_profile.json")
    eval_pairs = data_utils.load_json("/home/zfj/research-data/na-checking/aminer-new1/", "eval_na_checking_pairs_test.json")
    paper_dict = data_utils.load_json("./na-check-new", "paper_dict_used_mag_for_conna.json")
    name_aid_to_pids_new_test = dd(dict)

    for i, pair in enumerate(eval_pairs):
        if i % 10000 == 0:
            print("pair", i)
        aid = pair["aid1"]
        name = pair["name"]
        pid = pair["pid"]
        pid = str(pid)

        if pid not in paper_dict:
            continue
        for a_i, a in enumerate(paper_dict[pid].get("authors", [])):
            cur_a_name = a["name"]
            cur_a_name = string_utils.clean_name(cur_a_name)
            if cur_a_name == name:
                pid_order = str(pid) + "-" + str(a_i)
                if name_aid_to_pids_new_test.get(name, {}).get(aid, []):
                    name_aid_to_pids_new_test[name][aid].append(pid_order)
                else:
                    name_aid_to_pids_new_test[name][aid] = [pid_order]

    data_utils.dump_json(name_aid_to_pids_new_test, "./na-check-new", "test_author_pub_index_test.json")
    data_utils.dump_json(name_aid_to_pids_new, "./na-check-new", "test_author_pub_index_profile.json")


if __name__ == '__main__':
    # Processing raw data as follows to generate essential word embeddings.

    # gen_paper_dict_for_conna()
    # dump_pub_features_to_file()   # extract features of author name and words from publications
    # cal_feature_idf()                # calculate idf for each author name or word

    # emb_model = EmbeddingModel.Instance()
    # emb_model.train()                # train embeddings for author names and words

    # dump_feature_id_to_file()
    # gen_train_name_aid_to_pids()
    gen_test_name_aid_to_pids()
