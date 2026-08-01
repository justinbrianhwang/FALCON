# FALCON 실험 실행 가이드

실행에 필요한 내용만 담았습니다.

## 1. 환경 설치 (최초 1회)

```bash
conda env create -f environment.yml
conda activate falcon
pytest        # 전부 통과하면 환경 OK
```

## 2. 데이터 준비 (최초 1회)

아무 설정 없이 아래만 실행하면 `./data`에 자동 다운로드됩니다.

```bash
python scripts/prepare_data.py --datasets cifar10,cifar100,mnist,fmnist,svhn
```

완료되면 `./data/processed/<이름>.pkl`이 생기고, 실험 코드는 이 pkl만 읽습니다.

## 3. 실험 실행

```bash
# 합성 데이터 MVP 실행 (라운드별 정확도 출력)
python experiments/run_synthetic.py

# 전체 파이프라인: pair 검증 → 개입(restore/inject/sham) → 귀속 리포트
python -m falcon.reporting --runs-root . --reference ref_001 --failure fail_001 \
  --metric accuracy --output report.md
```

세부 실험 스크립트는 experiments/ 폴더에 추가될 예정이며, 그때마다 이 문서가 갱신됩니다.

## 4. 결과 보내기

실험이 끝나면 아래 한 줄 실행 후, 생성된 zip 파일을 보내주세요.

```bash
python scripts/collect_output.py
# -> tmp/Output_YYYY-MM-DD_HH-MM-SS.zip 생성됨
```

무거운 원시 텐서까지 필요하다고 요청받은 경우에만 `--full`을 붙여주세요.

## 문제 생기면

- `pytest` 실패: 파이썬/의존성 버전 문제일 가능성 — `conda env remove -n falcon` 후 1번부터 재시도.
- 데이터 다운로드 실패: 네트워크 문제 — 해당 데이터셋만 `--datasets cifar10`처럼 다시 실행 (이미 받은 것은 건너뜀).
