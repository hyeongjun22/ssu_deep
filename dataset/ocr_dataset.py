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

# --- 최종 수정된 OCR 데이터셋 클래스 ---
class OCRDataset:
    def __init__(self, image_paths_or_dir: any, label_dir: str, vocab_path: str, image_size: Tuple[int, int] = (128, 32), max_label_len: int = 25):
        print("🚀 OCR 데이터셋 초기화를 시작합니다...")
        
        # image_paths_or_dir가 디렉토리 경로인지, 파일 리스트인지 확인
        if isinstance(image_paths_or_dir, str) and os.path.isdir(image_paths_or_dir):
            # 디렉토리 경로가 주어지면, 해당 디렉토리 내의 모든 이미지 파일 경로를 리스트로 만듭니다.
            self.image_files = sorted([os.path.join(image_paths_or_dir, f) for f in os.listdir(image_paths_or_dir) if f.endswith(('.png', '.jpg'))])
        elif isinstance(image_paths_or_dir, list):
            # 파일 경로 리스트가 주어지면, 그대로 사용합니다.
            self.image_files = image_paths_or_dir
        else:
            raise ValueError("image_paths_or_dir는 디렉토리 경로(str) 또는 파일 경로 리스트(list)여야 합니다.")

        self.label_dir = label_dir
        self.image_size = image_size
        self.max_label_len = max_label_len
        
        self._load_vocab(vocab_path)
        self.vocab_size = len(self.char_to_id)
        self.pad_id = self.char_to_id['<PAD>']
        
        print(f"✅ 단어장 로드 완료. 총 글자 수: {self.vocab_size}")
        print(f"✅ 총 {len(self.image_files)}개의 이미지 파일을 처리 대상으로 설정했습니다.")

    def _load_vocab(self, vocab_path: str):
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"단어장 파일 '{vocab_path}'를 찾을 수 없습니다. 먼저 build_vocab.py를 실행하여 파일을 생성해주세요.")
        
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        self.char_to_id = vocab['char_to_id']
        self.id_to_char = {int(k): v for k, v in vocab['id_to_char'].items()}

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        # ✨✨✨ 핵심 수정 부분 ✨✨✨
        # self.image_files[idx]는 이제 '/content/drive/...' 와 같은 전체 경로를 담고 있습니다.
        image_path = self.image_files[idx]
        image_name = os.path.basename(image_path) # 경로에서 파일 이름만 추출
        # ✨✨✨✨✨✨✨✨✨✨✨✨✨
        
        label_name = os.path.splitext(image_name)[0] + '.json'
        label_path = os.path.join(self.label_dir, label_name)
        
        try:
            image = Image.open(image_path).convert('L')
            with open(label_path, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
                label = Label(**label_data)
        except (FileNotFoundError, ValidationError):
            return [], []

        word_images, word_labels = [], []
        for bbox in label.bbox:
            crop_box = (bbox.x[0], bbox.y[0], bbox.x[2], bbox.y[1])
            word_img = image.crop(crop_box).resize(self.image_size, Image.LANCZOS)
            
            word_img_np = np.array(word_img, dtype=np.float32) / 255.0
            word_images.append(word_img_np[np.newaxis, :])
            
            text = bbox.data
            label_seq = [self.char_to_id.get(char, self.pad_id) for char in text]
            
            padded_seq = np.full(self.max_label_len, self.pad_id, dtype=np.int32)
            seq_len = min(len(label_seq), self.max_label_len)
            padded_seq[:seq_len] = label_seq[:seq_len]
            word_labels.append(padded_seq)
            
        return word_images, word_labels

    def get_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_images, batch_labels = [], []
        while len(batch_images) < batch_size:
            idx = np.random.randint(0, len(self))
            images, labels = self[idx]
            if images:
                batch_images.extend(images)
                batch_labels.extend(labels)
        
        if not batch_images: # 만약 선택된 인덱스들에서 유효한 데이터를 하나도 못 찾았을 경우
            return np.array([]), np.array([])
            
        x_batch = np.array(batch_images[:batch_size], dtype=np.float32)
        t_batch = np.array(batch_labels[:batch_size], dtype=np.int32)

        return x_batch, t_batch