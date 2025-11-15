## ✅ Training Script Updated!

The training script now **automatically copies the best model** to `src/model.pt` when training completes.

---

## 🎯 What Changed

### 1. **Automatic Model Deployment**
After training finishes, the best model is automatically copied from:
- `training/models/nnue_best.pt` → `src/model.pt`

### 2. **Added import**
```python
import shutil  # For file copying
```

### 3. **Deployment Logic**
At the end of training:
```python
# Copy best model to src/ directory for deployment
best_model_path = os.path.join(output_dir, "nnue_best.pt")
src_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
deployment_path = os.path.join(src_dir, "model.pt")

if os.path.exists(best_model_path):
    os.makedirs(src_dir, exist_ok=True)
    shutil.copy2(best_model_path, deployment_path)
    print(f"\n✅ Copied best model to: {deployment_path}")
    print(f"   Ready for deployment!")
```

---

## 📝 Usage

### **Train normally:**
```bash
cd training
python3 train.py data.jsonl --max-lines 500000 --epochs 15
```

### **Output at end of training:**
```
============================================================
Training complete!
Best validation loss: 0.0234
============================================================

✅ Copied best model to: /path/to/magnus-carlsten/src/model.pt
   Ready for deployment!
```

---

## 📂 File Structure

```
magnus-carlsten/
├── training/
│   ├── models/
│   │   ├── nnue_best.pt      # Best model (source)
│   │   ├── nnue_epoch1.pt
│   │   └── nnue_epoch2.pt
│   └── train.py
│
└── src/
    ├── model.pt                # Deployed model (auto-copied)
    └── README.md               # Usage instructions
```

---

## 🔄 Manual Deployment (if needed)

### **Deploy a specific checkpoint:**
```bash
cd training
bash deploy_model.sh models/nnue_epoch10.pt
```

### **Deploy to custom location:**
```bash
cd training
bash deploy_model.sh models/nnue_best.pt ../custom/path/
```

### **Manual copy:**
```bash
cp training/models/nnue_best.pt src/model.pt
```

---

## 💻 Using the Deployed Model

### **Simple evaluation:**
```python
import torch
from training.nnue_model import SimpleNNUE
from training.features import board_to_features
import chess

# Load model
checkpoint = torch.load('src/model.pt', map_location='cpu')
model = SimpleNNUE(
    h1=checkpoint['config']['h1'],
    h2=checkpoint['config']['h2']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Evaluate
board = chess.Board()
features = board_to_features(board).unsqueeze(0)
with torch.no_grad():
    eval_cp = int(model(features).item() * checkpoint['config']['max_cp'])

print(f"Eval: {eval_cp:+d}cp")
```

### **In your chess engine:**
```python
# Import from src
import sys
sys.path.append('.')

from training.nnue_model import SimpleNNUE
from training.features import board_to_features

class Engine:
    def __init__(self):
        ckpt = torch.load('src/model.pt')
        self.model = SimpleNNUE(h1=ckpt['config']['h1'], h2=ckpt['config']['h2'])
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        self.max_cp = ckpt['config']['max_cp']
    
    def eval(self, board):
        features = board_to_features(board).unsqueeze(0)
        with torch.no_grad():
            return int(self.model(features).item() * self.max_cp)
```

---

## 🎯 Benefits

1. ✅ **No manual copying** - model automatically deployed after training
2. ✅ **Consistent location** - always in `src/model.pt`
3. ✅ **Ready for git** - src/model.pt can be tracked
4. ✅ **Simple imports** - clean path for your engine
5. ✅ **Backup maintained** - original still in `training/models/`

---

## 📊 Model Info

Check deployed model details:
```bash
cd src
python3 -c "
import torch
ckpt = torch.load('model.pt')
print(f'Epoch: {ckpt[\"epoch\"]}')
print(f'Val Loss: {ckpt[\"val_loss\"]:.4f}')
print(f'Config: {ckpt[\"config\"]}')
"
```

---

## ✨ Files Created

1. **`train.py`** (modified) - Auto-copy logic added
2. **`deploy_model.sh`** - Manual deployment script
3. **`src/README.md`** - Deployment documentation
4. **`MODEL_DEPLOYMENT.md`** - This file

---

**Your model is now ready to use in your chess engine!** 🚀

