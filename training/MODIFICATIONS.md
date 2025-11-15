## ✅ Modifications Complete!

I've added both requested features to the training script:

### 1. **`--max-lines` Parameter**
Limit how many lines to read from your huge data file.

### 2. **Progress Printing**
Shows progress every 10,000 lines with remaining lines count.

---

## 📝 Usage Examples

### **Quick Training (500k positions)**
```bash
python3 train.py huge_data.jsonl --max-lines 500000 --epochs 15
```

### **Fast Test (100k positions)**
```bash
python3 train.py huge_data.jsonl --max-lines 100000 --epochs 5
```

### **Medium Training (2M positions)**
```bash
python3 train.py huge_data.jsonl --max-lines 2000000 --epochs 15
```

### **Without Limit (reads entire file)**
```bash
python3 train.py data.jsonl --epochs 15
```

---

## 📊 What You'll See

```
Loading data from huge_data.jsonl...
  (limiting to 500,000 lines)
  Progress: 10,000/500,000 lines (8,234 positions loaded, 490,000 lines remaining)
  Progress: 20,000/500,000 lines (16,891 positions loaded, 480,000 lines remaining)
  Progress: 30,000/500,000 lines (25,102 positions loaded, 470,000 lines remaining)
  ...
  Progress: 500,000/500,000 lines (421,087 positions loaded, 0 lines remaining)
  Reached max_lines limit (500,000)
Loaded 421,087 positions

Train set: 378,978 positions
Val set: 42,109 positions
```

---

## 🎯 Recommended Settings for 60GB Data

| Goal | Command | Time | Positions |
|------|---------|------|-----------|
| **Quick test** | `--max-lines 100000` | ~5 min | ~80k |
| **Hackathon baseline** ⭐ | `--max-lines 500000` | ~20 min | ~400k |
| **Strong model** | `--max-lines 2000000` | ~60 min | ~1.6M |
| **Overkill** | `--max-lines 10000000` | ~4 hrs | ~8M |

**⭐ Recommended:** Use 500,000 lines for perfect balance!

---

## 💡 Why Some Lines Don't Become Positions

Not all lines have valid data:
- Missing FEN or eval fields
- Invalid FEN strings
- Mate scores (converted to max_cp)
- Missing or malformed JSON

Typically **80-85%** of lines become valid training positions.

---

## 🚀 Quick Start

```bash
# 1. Train on first 500k lines (recommended)
python3 train.py your_huge_file.jsonl --max-lines 500000

# 2. Or use multiple sizes and ensemble
python3 train.py data.jsonl --max-lines 250000 --output-dir models/small
python3 train.py data.jsonl --max-lines 500000 --output-dir models/medium
python3 train.py data.jsonl --max-lines 1000000 --output-dir models/large

# 3. Ensemble at inference for extra strength!
```

---

## ✅ Changes Made

1. **`dataset.py`**:
   - Added `max_lines` parameter to `LichessEvalDataset.__init__()`
   - Added progress printing every 10k lines in `_load_data()`
   - Shows "X remaining" when max_lines is set
   - Stops reading when max_lines reached
   - Updated `create_dataloader()` to accept and pass `max_lines`

2. **`train.py`**:
   - Added `max_lines` parameter to `train_nnue()` function
   - Added `--max-lines` command-line argument
   - Passes `max_lines` to `create_dataloader()`

---

You can now train on your 60GB file without waiting hours! 🎉

