# 동물 종류 판별 웹앱

FastAPI와 CPU용 ResNet-18 ImageNet 사전학습 모델을 사용하는 이미지 분류 웹앱입니다. 이미지 선택, 미리보기, 상위 3개 추정 결과와 신뢰도를 제공합니다. 결과는 전문적인 수의학적 판정이 아닌 참고용 AI 추정치입니다.

## 기술 스택

- Python 3.14 x64 (현재 환경 기준)
- FastAPI, Uvicorn
- Pillow
- PyTorch, Torchvision
- HTML, CSS, JavaScript

## 설치

Windows PowerShell에서 `upload` 폴더로 이동합니다.

```powershell
cd "C:\Users\songh\옾소\animal-classifier\upload"
python -m pip install -r requirements.txt
python -m pip install --index-url https://download.pytorch.org/whl/cpu -r requirements-model.txt
```

Python이 실제로 설치되어 있어야 합니다. `python`이 WindowsApps 실행 별칭으로만 연결되어 있다면 Python 3.12 x64를 설치하고 설치 과정에서 `Add Python to PATH`를 선택하세요.

## 모델 준비

별도의 모델 파일을 저장소에 복사하지 않습니다. 앱 첫 실행 시 `torchvision.models.ResNet18_Weights.DEFAULT`가 필요한 가중치를 자동으로 다운로드하고 로컬 Torch cache에 저장합니다. 일반 노트북 CPU로 실행할 수 있으며 GPU는 필요하지 않습니다. 최초 실행 시 인터넷 연결과 약 45MB의 모델 다운로드가 필요합니다.

## 실행

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

브라우저 주소:

```text
http://127.0.0.1:8001
```

이미지 선택 → 미리보기 확인 → `이미지 분석하기`를 누르면 상위 3개 추정 결과를 확인할 수 있습니다. API 상태는 <http://127.0.0.1:8001/api/health>에서 확인합니다.

## 보안 및 저장소 정책

이 폴더에는 인증정보, 개인정보, 테스트 이미지, 로그, 캐시, 임시 파일, 대용량 모델 가중치를 포함하지 않습니다. 모델 cache는 사용자 컴퓨터에만 생성되며 저장소에 올리지 않습니다.

