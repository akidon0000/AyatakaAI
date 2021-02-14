# -------------------------------------
# ラベリングによる学習/検証データの準備
# -------------------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# from tensorflow._api import keras
# from tensorflow.keras.utils import np_utils
# from tensorflow.keras import optimizers as opt
# from tensorflow.keras import models as mdl, layers as ly
# from keras.utils import np_utils
# from keras import optimizers as opt
# from keras import models as mdl, layers as ly

from PIL import Image
# import os
import glob
import numpy as np
import random
import math

# 画像が保存されているルートディレクトリのパス
root_dir = "/Users/akidon/MacMini/Program-mini/AyatakaAI_py/images"
# 商品名
categories = ["orange", "lemon", "apple"]

# 画像データ用配列
X = []
# ラベルデータ用配列
Y = []

# 画像データごとにadd_sample()を呼び出し、X,Yの配列を返す関数


def make_sample(files):
    global X, Y
    X = []
    Y = []
    for category_index, file_name in files:
        add_sample(category_index, file_name)
    return np.array(X), np.array(Y)

# 渡された画像データを読み込んでXに格納し、
# 画像データに対応するcategoryのidxをYに格納する関数


def add_sample(category_index, file_name):
    img = Image.open(file_name)
    img = img.convert("RGB")
    img = img.resize((150, 150))
    data = np.asarray(img)
    X.append(data)
    Y.append(category_index)


# 全データ格納用配列
allfiles = []

# カテゴリ配列の各値と、それに対応するidxを認識し、全データをallfilesにまとめる
for index, category in enumerate(categories):
    image_dir = root_dir + "/" + category
    files = glob.glob(image_dir + "/*.jpg")
    for f in files:
        allfiles.append((index, f))

# シャッフル後、学習データと検証データに分ける
random.shuffle(allfiles)
number_of_train = math.floor(len(allfiles) * 0.8)
train = allfiles[0:number_of_train]
test = allfiles[number_of_train:]
X_train, Y_train = make_sample(train)
X_test, Y_test = make_sample(test)
xy = (X_train, Y_train, X_test, Y_test)
# データを保存する（データの名前を「fruit_data.npy」としている
np.save(str(root_dir + "/fruit_data.npy"), xy)

# -------------------------------------
# モデルの構築
# -------------------------------------

# model = mdl.Sequential()
# model.add(ly.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)))
# model.add(ly.MaxPooling2D(2, 2))
# model.add(ly.Conv2D(64, (3, 3), activation="relu"))
# model.add(ly.MaxPooling2D(2, 2))
# model.add(ly.Conv2D(128, (3, 3), activation="relu"))
# model.add(ly.MaxPooling2D(2, 2))
# model.add(ly.Conv2D(128, (3, 3), activation="relu"))
# model.add(ly.MaxPooling2D(2, 2))
# model.add(ly.Flatten())
# model.add(ly.Dense(512, activation="relu"))
# model.add(ly.Dense(10, activation="sigmoid"))  # 分類先の種類分設定
model = keras.models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(10, activation="sigmoid"))  # 分類先の種類分設定

# モデル構成の確認
model.summary()


# -------------------------------------
# モデルのコンパイル
# -------------------------------------

model.compile(
    loss="binary_crossentropy",
    # optimizer=opt.RMSprop(lr=1e-4),
    optimizer=keras.optimizers.RMSprop(lr=1e-4),
    metrics=["acc"])

# -------------------------------------
# データの準備
# -------------------------------------

nb_classes = len(categories)

X_train, X_test, Y_test, Y_test = np.load(str(root_dir + "/fruit_data.npy"))

# データの正規化
X_train = X_train.asType("float") / 255
X_test = X_test.asType("float") / 255

# kerasで扱えるようにcategoriesをｗベクトルに変換
# Y_train = np_utils.to_categorical(Y_train, nb_classes)
# Y_test = np_utils.to_categorical(Y_test, nb_classes)
Y_train = keras.utils.to_categorical(Y_train, nb_classes)
Y_test = keras.utils.to_categorical(Y_test, nb_classes)

# モデルの学習
model = model.fit(
    X_train,
    Y_train,
    epochs=10,
    batch_size=6,
    validation_data=(X_test, Y_test))

print("finish")
