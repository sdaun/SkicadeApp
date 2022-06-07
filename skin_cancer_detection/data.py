import numpy as np
import pandas as pd
from PIL import Image
import pickle5 as pickle
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def get_data(resize_width, resize_height):
    skin_df = pd.read_csv('../raw_data/HAM10000_metadata.csv')
    skin_df['path'] = [f'../raw_data/HAM10000_all/{img}.jpg' for img in skin_df['image_id']]
    skin_df['image_resized'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((resize_width,resize_height))))
    return skin_df

def save_df_as_pickle(skin_df, name_of_pickle, path):
    skin_df.to_pickle(f"{path}/{name_of_pickle})")

def get_data_from_pickle(name_of_pickle, path):
    with open(f"{path}/{name_of_pickle})", "rb") as fh:
        skin_df = pickle.load(fh)
    return skin_df


def data_preparation(skin_df, val_set = False):
    y = skin_df['dx']
    dict_target = {'bkl':0, 'nv':1, 'df':2, 'mel':3, 'vasc':4, 'bcc':5, 'akiec':6}
    y_num = y.map(dict_target.get)
    skin_df['target'] = y_num
    y_cat = to_categorical(y_num, num_classes = 7)
    X = skin_df['image_resized']
    if val_set:
        X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        X_train_stack = np.stack(X_train)
        X_test_stack = np.stack(X_test)
        X_val_stack = np.stack(X_val)
        return X_train_stack, X_test_stack, X_val_stack, y_train, y_test, y_val
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.3, random_state=42)
        X_train_stack = np.stack(X_train)
        X_test_stack = np.stack(X_test)
        return X_train_stack, X_test_stack, y_train, y_test


if __name__ == '__main__':
    skin_df = get_data(100,75)
    X_train_stack, X_test_stack, y_train, y_test = data_preparation(skin_df, val_set = False)
    print('done')
