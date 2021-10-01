# NLP_Text-categorization

這次分享的是在前陣子競賽 以及資策會期末專題中使用的NLP相關套件 <br> 
*SimpleTransformers*
基於Transformers所改寫的簡易版套件<br> 
可以簡單的導入以及驗證訓練模型 <br> 


# 可利用標籤後的資料 進行訓練  <br>讓模型可以依照標籤的規則去判斷結果

預訓練模型好處就是可以省去非監督式學習的時間<br> 
可以讓後續下游任務進行得更快<br> 
訓練後的模型 會依照你所給予的標籤 進行判斷<br> 

例如你在送進去大量關於"稱讚" 以及 "責罵" 的句子後 <br> 
你在送入新的句子給經過訓練後的model後  <br>
他就可以判斷出你這句子的語意是在稱讚或是責罵 <br>
當然 準確度會依照你送入的資料質量產生波動  <br>


# 注意事項

如果你有NVIDIA的顯示卡 這樣會事半功倍 <br>
請先下載CUDA相關套件 以及pytorch <br>
要注意 CUDA <=> pytorch 之間有支援性問題 <br> 

# 使用方式-1 

附件的rar裡面 有兩個csv  <br>
分別是train.csv 以及 test.csv  <br>
放置到跟PY檔案同目錄下即可  <br>

# 使用方式-2

在 **go = MachineLearning('roberta', 'roberta-base', 38, 60)**   進行設定 <br>

<1> 'roberta', 'roberta-base' <br>

roberta 是預訓練模型 <br>
除了roberta外 還有其他的預訓練模型可供選擇  <br>
roberta-base 則是他的子體  <br>
子體簡單來說就是會有一些小功能的改變  <br>
比如說他們訓練的語料 如英文\小寫英文\不分大小寫的英文\中文\他國語言等  <br>
可以依照自己的任務需求 去選擇自己最是適合的模型  <br>

<2> 38, 60 <br>

這兩個數字分別代表超參數 batch_size 以及 epoch <br>
<br>
batch_size 簡單來說 就是程式一次會拿多少資料進去搞東搞西 <br>
這數字受到硬體限制 <br>
越大的話 所需要的GPU 記憶體越多 <br>
我的顯示卡是2080 Super 38左右就是極限 再往上去會報錯<br>

epoch就是讀完全部訓練資料一次的次數<br>
如同上面的batch_size <br>
如果今天訓練10筆資料 batch_size = 1 epoch = 2 <br>
這樣執行時 他一次會拿一筆資料 共拿十次 <br>
拿完十次後 epoch+1 開始進行第二次"拿十次"的動作<br>
當 epoch=2時 訓練就完成了!<br>







