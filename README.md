# nishika_test1_price_of_old_apartment
 
Nishikaのトレーニングコンペ【中古マンション価格予測】に挑戦したのでその復習
lightgbmを用いたデータ分析

## 基本的にCRISP-DMに沿って予測を実施
CRISP-DM実施手順
1)Business Understanding
2)Data Understanding
3)Data Preparation
4)Modeling
5)Evaluation
6)Deployment

### 今回はData Preparationから

#### EDA(探索的データ分析)を実施（eda.ipynb）→Modelの作成(model.ipynb)

1)必要なライブラリをimportする。
glob,pandas,numpy,matplotlib.pyplot,seaborn

2)テストデータを結合する。

3)不要なカラムを削除、データ型の変換を実施する。

４）Modelを作成する。(lightgbmを使用)

5)作成したモデルを使って検証データの結果を予測




