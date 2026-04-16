import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import os
import joblib
import warnings

# 忽略因為套件版本導致的警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. 神經網路模型架構先備定義 (來自三個 ipynb)
# ==========================================

class TextCNN(nn.Module):
    """
    來自 CNNeo_CNN_BioBERT.ipynb 的文字卷積網路模型
    預設 BioBERT 特徵大小為 768
    """
    def __init__(self, input_size=768, num_filters=120, filter_sizes=[3, 4, 5], output_size=2, dropout=0.2):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, input_size)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in x]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        return self.fc(x)

class FullyConnectedModel(nn.Module):
    """
    來自 CNNeo_FCNN_BioBERT 與 CNNeo_FCN_TF.ipynb 的全連接網路模型
    透過 dropout_p 參數控制差異 (TF 為 0.2, BioBERT 為 0.5)
    """
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.5):
        super(FullyConnectedModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.drop = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        return self.fc2(x)

# ==========================================
# 2. 資料前處理輔助函式
# ==========================================

def trans_Mutated(x):
    """將 Mutated Peptide 補上 X 直到長度達到 11"""
    x = str(x)
    return x + 'X' * max(0, 11 - len(x))

def getKmers(sequence_str, size):
    """將字串進行 K-mers 切割"""
    return ' '.join([sequence_str[i:i+size].lower() for i in range(len(sequence_str) - size + 1)])

# ==========================================
# 3. 三個模型的推論 (Inference) 實作
# ==========================================

def infer_CNN_BioBERT(df, model_path, device):
    """推論 CNNeo_CNN_BioBERT"""
    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
    biobert_model = AutoModel.from_pretrained('dmis-lab/biobert-base-cased-v1.1').to(device)
    biobert_model.eval()

    df['M'] = df['peptide'].apply(trans_Mutated)
    df['trans'] = df.apply(lambda row: getKmers(row['hla'] + row['M'], 4), axis=1)

    input_ids, attention_masks = [], []
    for text in df['trans']:
        encoded = tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=64,
            padding='max_length', return_tensors='pt', truncation=True
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    
    input_ids = torch.cat(input_ids, dim=0).to(device)
    attention_masks = torch.cat(attention_masks, dim=0).to(device)

    with torch.no_grad():
        features = biobert_model(input_ids, attention_mask=attention_masks).last_hidden_state
    
    model = TextCNN(input_size=768, num_filters=120, filter_sizes=[3,4,5], output_size=2, dropout=0.2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    scores = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            batch = features[i:i+batch_size]
            outputs = model(batch)
            proba = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            scores.extend(proba)
    return scores

def infer_FCNN_BioBERT(df, model_path, device, scaler_path=None, hla_pseudo_dict=None):
    """推論 CNNeo_FCNN_BioBERT"""
    tokenizer = AutoTokenizer.from_pretrained('monologg/biobert_v1.1_pubmed')
    
    # 如果有提供 pseudo sequence 的對照表就轉換，不然預設使用原字串
    if hla_pseudo_dict is None: hla_pseudo_dict = {}
    df['hla_seq'] = df['hla'].apply(lambda x: hla_pseudo_dict.get(x, x))
    df['M'] = df['peptide'].apply(trans_Mutated)
    df['trans'] = df.apply(lambda row: getKmers(row['hla_seq'] + row['M'], 2), axis=1)
    
    input_ids = []
    for text in df['trans']:
        encoded = tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=65,
            padding='max_length', return_tensors='pt', truncation=True
        )
        input_ids.append(encoded['input_ids'])
    
    input_ids_np = torch.cat(input_ids, dim=0).cpu().numpy()
    
    # 根據原始模型架構，如果有 scaler 必須進行標準化
    if scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        x_scale = scaler.transform(input_ids_np)
    else:
        x_scale = input_ids_np

    X = torch.tensor(x_scale, dtype=torch.float32).to(device)
    input_size = X.shape[1] 
    
    model = FullyConnectedModel(input_size=input_size, hidden_size=53, output_size=2, dropout_p=0.5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    scores = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            outputs = model(X[i:i+batch_size])
            proba = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            scores.extend(proba)
    return scores

def infer_FCN_TF(df, model_path, device, tfidf_path=None):
    """推論 CNNeo_FCN_TF"""
    df['M'] = df['peptide'].apply(trans_Mutated)
    df['trans'] = df.apply(lambda row: getKmers(row['hla'] + row['M'], 6), axis=1)

    if not tfidf_path or not os.path.exists(tfidf_path):
        raise FileNotFoundError(f"[錯誤] 請提供 TF-IDF Vectorizer 檔案 ({tfidf_path})，否則 TF 模型無法轉換文字。")
    
    vectorizer = joblib.load(tfidf_path)
    x_scale = vectorizer.transform(df['trans']).toarray()
    X = torch.tensor(x_scale, dtype=torch.float32).to(device)
    input_size = X.shape[1] 
    
    model = FullyConnectedModel(input_size=input_size, hidden_size=64, output_size=2, dropout_p=0.2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    scores = []
    with torch.no_grad():
        outputs = model(X)
        proba = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        scores.extend(proba)
    return scores

# ==========================================
# 4. 總管 API：推論並組裝回 DataFrame
# ==========================================

def run_cnneopp_pipeline(input_df, model_dir='models'):
    """
    輸入: df 至少包含 `hla` 與 `peptide` 兩個欄位
    輸出: 原 df 加上蒐集到的所有的 model 預測分數 (cnneopp_score)
    """
    df = input_df.copy()
    
    # 格式清理
    df['hla'] = df['hla'].astype(str).str.replace('*', '').str.replace(' ', '').str.replace('\xa0', '')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Info] 使用裝置: {device}")
    
    cnn_biobert_path = os.path.join(model_dir, 'CNN_BioBERT.pth')
    fcnn_biobert_path = os.path.join(model_dir, 'FCNN_BioBERT.pth')
    fcn_tf_path = os.path.join(model_dir, 'FCNN_TF.pth')
    
    # 注意：這些是由你在原本 notebook 中「訓練時」必須主動用 joblib.dump 儲存的狀態，
    # 若無儲存將引發例外。
    tfidf_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
    scaler_path = os.path.join(model_dir, 'fcnn_scaler.pkl')
    
    models_found = 0
    
    # 1. CNN_BioBERT 執行
    if os.path.exists(cnn_biobert_path):
        print("[Execute] 執行推論 CNN_BioBERT 模型...")
        df['score_CNN_BioBERT'] = infer_CNN_BioBERT(df, cnn_biobert_path, device)
        models_found += 1
    else:
        print(f"[Skip] 找不到模型路徑 {cnn_biobert_path}")
        
    # 2. FCNN_BioBERT 執行
    if os.path.exists(fcnn_biobert_path):
        print("[Execute] 執行推論 FCNN_BioBERT 模型...")
        # 這裡的 hla_pseudo_dict 必須要提供你 MHC_pseudo.dat 解析出 dict，這裡我們先用空白代替
        df['score_FCNN_BioBERT'] = infer_FCNN_BioBERT(df, fcnn_biobert_path, device, scaler_path, hla_pseudo_dict=None)
        models_found += 1
    else:
        print(f"[Skip] 找不到模型路徑 {fcnn_biobert_path}")

    # 3. FCN_TF 執行
    if os.path.exists(fcn_tf_path):
        print("[Execute] 執行推論 FCN_TF 模型...")
        try:
             df['score_FCN_TF'] = infer_FCN_TF(df, fcn_tf_path, device, tfidf_path)
             models_found += 1
        except FileNotFoundError as e:
             print(e)
    else:
        print(f"[Skip] 找不到模型路徑 {fcn_tf_path}")
        
    print(f"\n[Info] 總計載入並取得分數的模型數量: {models_found}\n")

    # 清除非必要的暫存特徵欄位
    tmp_cols = ['M', 'trans', 'hla_seq']
    for col in tmp_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df

if __name__ == "__main__":
    # ===== 使用範例 =====
    test_data = pd.DataFrame({
        'hla': ['A*02:01', 'B*07:02'],
        'peptide': ['SLYNTVATL', 'TPRVTGGGAM']
    })
    
    print("--- 原始 Dataframe ---")
    print(test_data)
    print("\n--- 執行推論中 ---")
    
    try:
        final_df = run_cnneopp_pipeline(test_data, model_dir='models')
        print("--- 最終結果 Dataframe ---")
        print(final_df)
    except Exception as e:
        print(f"執行時發生錯誤: {e}")
