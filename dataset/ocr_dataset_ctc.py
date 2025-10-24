import os
import json
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple
from pydantic import BaseModel, ValidationError

# --- Pydantic 모델 정의 (기존과 동일) ---
class Annotation(BaseModel):
    object_recognition: int
    text_language: int

class Dataset(BaseModel):
    category: int
    identifier: str
    label_path: str
    name: str
    src_path: str
    type: int

class Images(BaseModel):
    acquistion_location: str
    application_field: str
    background: int
    data_captured: str
    height: int
    identifier: str
    media_type: int
    pen_color: str
    pen_type: int
    type: str
    width: int
    writer_age: int
    writer_sex: int
    written_content: int

class BBox(BaseModel):
    data: str
    id: int
    x: List[int]
    y: List[int]

class Label(BaseModel):
    Annotation: Annotation
    Dataset: Dataset
    Images: Images
    bbox: List[BBox]

# --- CTC 학습용으로 수정된 OCR 데이터셋 클래스 ---
class OCRDataset:
    def __init__(self, image_paths_or_dir: any, label_dir: str, vocab_path: str, image_size: Tuple[int, int] = (64, 256), max_label_len: int = 25):
        print("🚀 OCR 데이터셋 초기화를 시작합니다...")
        self.image_dir = None
        if isinstance(image_paths_or_dir, str):
            self.image_dir = image_paths_or_dir
            self.image_paths = [os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        else:
            self.image_paths = image_paths_or_dir

        self.label_dir = label_dir
        self.image_size = image_size
        self.max_label_len = max_label_len

        # 단어장 로드
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        self.char_to_id = self.vocab['char_to_id']
        self.id_to_char = self.vocab['id_to_char']
        self.vocab_size = len(self.char_to_id)
        self.pad_id = self.char_to_id['<PAD>']
        print(f"✅ 단어장 로드 완료. 총 글자 수: {self.vocab_size}")
        print(f"✅ 총 {len(self.image_paths)}개의 이미지 파일을 처리 대상으로 설정했습니다.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]: # ✨ 반환 타입에 List[int] 추가
        img_path = self.image_paths[idx]
        base_filename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, f"{base_filename}.json")

        try:
            image = Image.open(img_path).convert('L') # Grayscale로 변환
            with open(label_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                label = Label(**data)
        except (FileNotFoundError, ValidationError, json.JSONDecodeError, OSError):
            return [], [], [] # ✨ 반환값 개수 3개로 수정

        word_images, word_labels, word_label_lengths = [], [], [] # ✨ 라벨 길이 저장을 위한 리스트 추가
        for bbox in label.bbox:
            # BBox 좌표로 이미지 크롭 및 리사이즈
            crop_box = (bbox.x[0], bbox.y[0], bbox.x[2], bbox.y[1])
            word_img = image.crop(crop_box).resize(self.image_size, Image.LANCZOS)
            
            # 이미지를 NumPy 배열로 변환하고 정규화
            word_img_np = np.array(word_img, dtype=np.float32) / 255.0
            word_images.append(word_img_np[np.newaxis, :])
            
            # 텍스트를 ID 시퀀스로 변환
            text = bbox.data
            label_seq = [self.char_to_id.get(char, self.pad_id) for char in text]
            
            # ✨ 패딩 전 실제 길이 저장
            seq_len = min(len(label_seq), self.max_label_len)
            word_label_lengths.append(seq_len)
            
            # 시퀀스 패딩
            padded_seq = np.full(self.max_label_len, self.pad_id, dtype=np.int32)
            padded_seq[:seq_len] = label_seq[:seq_len]
            word_labels.append(padded_seq)
            
        return word_images, word_labels, word_label_lengths # ✨ 3개 값 반환

    def get_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: # ✨ 반환 타입 수정
        batch_images, batch_labels, batch_label_lengths = [], [], [] # ✨ 라벨 길이 저장을 위한 리스트 추가
        
        while len(batch_images) < batch_size:
            idx = np.random.randint(0, len(self))
            images, labels, lengths = self[idx] # ✨ 3개 값 받기
            
            if images:
                batch_images.extend(images)
                batch_labels.extend(labels)
                batch_label_lengths.extend(lengths) # ✨ 배치 리스트에 길이 추가

        # 최종 배치 크기에 맞게 자르기
        x_batch = np.array(batch_images[:batch_size], dtype=np.float32)
        t_batch = np.array(batch_labels[:batch_size], dtype=np.int32)
        t_len_batch = np.array(batch_label_lengths[:batch_size], dtype=np.int32) # ✨ 길이 배치 생성

        return x_batch, t_batch, t_len_batch # ✨ 3개 값 반환