# PSO_Car_simulation

## 程式碼說明

### DataPoint.py
此份程式碼設計沿用作業一、二的資料結構，去除不需要的部分，另定義 PSO class 作為 data structurer 使用，class 中存有 x、pi 與 v 的資訊，並設計一個 Best_PSO 儲存最佳解 pg，這兩個 class 同時有 update function，用 deepcopy更新參數，避免記憶體位址的混亂。

### PSO.py
PSO算法實作部分，主要是按照左邊演算法實作，RBFN 的計算基本上就是直接用上份作業沒有改。PSO 演算法內容，定義兩個 function：initial 與 update_pos。
左方演算法 I 從 1 開始，故 initial 是處理 i=0  的粒子群狀態，包括計算 pi、fitness、update x、v 這幾樣項目。
Update_pos 則是實現整個粒子群的 x、v 2的更新。以及限定移動範圍（即演算法最下方的 for 迴圈）。pso_compute()為主要流程控制的 function，前端介面會呼叫這個 function 並傳入指定參數，在 function 中回直接計算 fitness，並用 sorting的方式找出 pg ，最後 update_pos更新。
### UI.py 、 draw.py
同樣沿用上次作業，最後在 GUI 與 draw 兩個介面中設計界面與畫圖，其中 draw.py 包含畫軌道與車子移動路線，GUI 則是讓使用者可以自行輸入參數。

## 簡單整理：
這次作業相較於基因演算法，我覺得計算上快很多，實作起來也比較簡單。在 fitness 與 error_rate 部分基本上每次都是收斂到差不多的數值，theta1、theta2 跟 v_max 主要是影響浮動的大小，但收斂速度上我覺得好像沒有到差太大，另外迭代次數調大，基本上都可以順利收斂。
