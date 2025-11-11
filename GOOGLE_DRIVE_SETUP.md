# Hướng dẫn cấu hình Google Drive để tải model weights

## Bước 1: Upload model lên Google Drive

1. Mở Google Drive của bạn
2. Upload file `best.pt` (model weights) lên Drive
3. Right-click vào file → Chọn **Share** (Chia sẻ)
4. Chọn **Anyone with the link** (Bất kỳ ai có link)
5. Copy link chia sẻ

## Bước 2: Lấy File ID từ link

Link chia sẻ sẽ có dạng:
```
https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/view?usp=sharing
```

File ID là phần giữa `/d/` và `/view`:
```
1AbCdEfGhIjKlMnOpQrStUvWxYz
```

## Bước 3: Tạo file `.env`

Copy file `.env.example` thành `.env` và đặt `GDRIVE_FILE_ID`:

```
GDRIVE_FILE_ID=1AbCdEfGhIjKlMnOpQrStUvWxYz
```

## Bước 4: Chạy setup

```powershell
python setup.py
```

Script sẽ tự động tải model từ Google Drive về folder `weights/` (nếu `GDRIVE_FILE_ID` được cấu hình đúng)

---

## Alternative: Manual Download (Nếu tự động không hoạt động)

Nếu việc tải tự động gặp lỗi, bạn có thể:

1. Tải file `best.pt` từ Google Drive về máy
2. Tạo folder `weights/` trong project
3. Copy file `best.pt` vào folder `weights/`

```
for_github_repo/
└── weights/
    └── best.pt
```

Sau đó chạy:
```bash
python run.py your_image.jpg
```

---

## Kiểm tra model đã tải thành công

```bash
# Windows
dir weights\best.pt

# Linux/Mac  
ls -lh weights/best.pt
```

Nếu thấy file với size ~6MB là đã thành công!
