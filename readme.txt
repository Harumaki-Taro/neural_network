## このプログラムについて

深層学習のための練習用のライブラリです。

密結合層、畳み込み層、ドロップアウト層などを自由に連結させてニューラルネットをつくり、学習させることができます。

様々な画像データセットでの学習を想定して作ってありますが 、あくまで練習用のためデータセットとしては

- iris (https://www.kaggle.com/uciml/iris)
- mnist (http://yann.lecun.com/exdb/mnist/)

のみ用意してあります。

`mnist_cnn.cpp`（密結合ニューラルネット）や `mnist_fnn.cpp`（畳み込みニューラルネット）に使用例が記載されています。



参考にしたページ
A Neural Network in 10 lines of C++ Code
https://cognitivedemons.wordpress.com/2017/07/06/a-neural-network-in-10-lines-of-c-code/
