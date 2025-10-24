import os
import json
from tqdm import tqdm
from pydantic import BaseModel, ValidationError
from typing import List

# ---------------------------------------------------------------------------
# 단어장 생성에 필요한 최소한의 Pydantic 모델 정의 (기존과 동일)
# ---------------------------------------------------------------------------
class BBox(BaseModel):
    data: str

class LabelForVocab(BaseModel):
    bbox: List[BBox]

# ---------------------------------------------------------------------------
# CTC용 단어장 생성 및 저장 함수
# ---------------------------------------------------------------------------
def build_and_save_vocab_ctc(label_dir: str, output_file: str):
    """
    지정된 디렉토리의 모든 JSON 라벨 파일을 읽어 CTC 학습에 필요한
    <PAD>와 <BLANK> 토큰이 포함된 단어장을 만들고 파일로 저장합니다.
    """
    if not os.path.exists(label_dir):
        print(f"🚨 오류: '{label_dir}' 경로를 찾을 수 없습니다.")
        print("라벨 파일이 있는 디렉토리 경로를 올바르게 지정해주세요.")
        return

    print(f"'{label_dir}' 디렉토리에서 단어장 생성을 시작합니다...")

    char_set = set()
    all_label_files = os.listdir(label_dir)

    for label_file in tqdm(all_label_files, desc="라벨 파일 분석 중"):
        json_path = os.path.join(label_dir, label_file)
        
     
        if not label_file.endswith('.json'):
            continue
            
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                label = LabelForVocab(**data)
                for item in label.bbox:
                    char_set.update(item.data)
        except (json.JSONDecodeError, ValidationError, FileNotFoundError):
            # 오류가 있는 파일은 건너뜁니다.
            continue

    # ✨✨✨ CTC를 위한 핵심 수정 부분 ✨✨✨
    # 1. 기존 문자들을 정렬합니다.
    char_list = sorted(list(char_set))
    
    # 2. '<PAD>' 토큰을 맨 앞에, '<BLANK>' 토큰을 맨 뒤에 추가합니다.
    #    - <PAD>는 주로 0번 인덱스에 위치시킵니다.
    #    - <BLANK>는 CTC Loss에서 특별한 역할을 하며, 보통 마지막 인덱스에 둡니다.
    special_tokens = ['<PAD>']
    char_list = special_tokens + char_list
    char_list.append('<BLANK>')
    # ✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨

    char_to_id = {char: i for i, char in enumerate(char_list)}
    id_to_char = {i: char for i, char in enumerate(char_list)}

    # 생성된 단어장을 파일로 저장
    vocab_data = {'char_to_id': char_to_id, 'id_to_char': id_to_char}
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=4)

    print(f"\\n✅ CTC용 단어장 생성 완료! 총 {len(char_to_id)}개의 고유 문자를 '{output_file}'에 저장했습니다.")
    print(f"  - PAD 토큰 ('<PAD>') ID: {char_to_id['<PAD>']}")
    print(f"  - BLANK 토큰 ('<BLANK>') ID: {char_to_id['<BLANK>']}")


# ---------------------------------------------------------------------------
# 스크립트 실행 부분
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # ⚠️ 사용자의 로컬 환경에 맞게 경로를 수정해주세요!
    # 예: 'D:/my_project/dataset/train_labels'
    LABEL_FILES_DIR = "train_labels" 
    
    # 생성된 vocab_ctc.json 파일을 저장할 경로
    VOCAB_SAVE_PATH = "vocab_ctc.json"

    # 실행!
    build_and_save_vocab_ctc(LABEL_FILES_DIR, VOCAB_SAVE_PATH)